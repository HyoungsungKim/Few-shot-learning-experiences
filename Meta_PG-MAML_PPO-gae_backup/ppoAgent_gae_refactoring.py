import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

GAMMA = 0.99
GAE_LAMBDA = 0.95


class env:
    def __init__(self, env_states_list, env_labels_list):
        self.states = env_states_list
        self.labels = env_labels_list
        self.first_states = None
        self.first_labels = None
       # print(np.array(self.states).shape)
       # print(np.array(self.labels))
       # assert 1 == 2
       
    def __len__(self):
        return len(self.states)
        
    def get_init_states_and_labels(self):
        self.first_states = self.states[0]
        self.first_labels = self.labels[0]
        
        return self.first_states, self.first_labels
        
    def step(self, actions):
        rewards = 0
        is_done = None
        current_states = None
        current_labels = None
        next_states = None
        next_labels = None
        
        if self.first_states is not None:
            current_states = self.first_states
            current_labels = self.first_labels
            self.first_states = None
            self.first_labels = None
        else:
            current_states = self.states[0]
            current_labels = self.labels[0]
        
        if len(self.states) <= 1:            
            rewards = [1 if actions[j] == current_labels[j] else 0 for j in range(len(current_labels))]
            is_done = True
            next_states = None
            next_labels = None
        else:            
            rewards = [1 if actions[j] == current_labels[j] else 0 for j in range(len(current_labels))]
            is_done = False
            next_states = self.states[1]
            next_labels = self.labels[1]
        
        if len(self.states) >= 1:
            del self.states[0]
            del self.labels[0]

        '''    
        print(sum(rewards)/len(current_labels))
        print(actions)
        print(current_labels)
        assert 1 == 2
        
        print(next_states.shape)
        print(next_labels.shape)

        print(current_state.shape)
        print(current_labels.shape)
        
        assert 1 == 2
        '''
        return next_states, next_labels, rewards, is_done, current_states
        

class PPOAgent:
    def __init__(self, model, model_optim,
                 gamma, entropy_weight, 
                 class_num, device):
        self.model = model
        self.model_optim = model_optim
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.device = device
        
        self.model_fast_weight = OrderedDict(model.named_parameters())
        
        self.transition = list()
        self.predicted_reward = 0
        self.total_reward = 0
        self.inner_lr = 0.01
        
        self.old_log_prob = None
                
    def _reset(self):
        self.total_reward = 0
        
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, gae_lambda=GAE_LAMBDA):                
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = (rewards[step] + gamma * values[step + 1] * masks[step]) - values[step]
            gae = delta + gamma * gae_lambda * masks[step] * gae
            returns.insert(0, gae+values[step])

        return returns
    
    def select_action(self, states, labels):
        dist, critic_logit = self.model(states)
        selected_action = dist.sample()
        log_prob = dist.log_prob(selected_action)
        entropy = dist.entropy().mean()
        
        _, predict_labels = torch.max(critic_logit.data, 1)
        assert len(predict_labels) == len(labels)
        predicted_value = [1 if predict_labels[j] == labels[j] else 0 for j in range(len(labels))]
        predicted_value_t = torch.tensor([predicted_value], dtype=torch.float, requires_grad=True).to(self.device)
        
        self.transition = [states, predicted_value_t, log_prob, entropy]
        
        return selected_action
    
    def step(self, action, env):
        next_states, next_labels, reward, done, _ = env.step(action)
        self.transition.extend([next_states, next_labels, reward, done])
        
        return next_states, next_labels, reward, done
    
    def ppo_inner_update(self, states, labels_t, actions, log_probs, returns, values_t, clip_param):
        for state, label_t, action, log_prob, return_, value in zip(states, labels_t, actions, log_probs, returns, values_t):
            dist, critic_logit = self.model(state)
            _, predict_labels = torch.max(critic_logit.data, 1)
            assert len(predict_labels) == len(label_t)
            predicted_value = [1 if predict_labels[j] == label_t[j] else 0 for j in range(len(label_t))]
            predicted_value_t = torch.tensor([predicted_value], dtype=torch.float, requires_grad=True).to(self.device)

            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(action)
            
            ratio = (new_log_prob - log_prob).exp()
                    
            advantage = return_ - value
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            
            regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
            value_loss = nn.MSELoss()(return_, predicted_value_t) + regularizer
                    
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss += -self.entropy_weight * entropy.mean()
            
            loss = value_loss + policy_loss 
            inner_value_gradients = torch.autograd.grad(loss, self.model_fast_weight.values(), create_graph=True, allow_unused=True)
            
            self.model_fast_weight = OrderedDict(
                (name, param - self.inner_lr * (0 if grad is None else grad))                    
                for ((name, param), grad) in zip(self.model_fast_weight.items(), inner_value_gradients)                    
            )
        
        
    def ppo_update(self, states, labels_t, actions, log_probs, returns, values_t, clip_param, loss_list):
        self.model.weight = self.model_fast_weight
        for state, label_t, action, log_prob, return_, value in zip(states, labels_t, actions, log_probs, returns, values_t):
            dist, critic_logit = self.model(state)
            _, predict_labels = torch.max(critic_logit.data, 1)
            assert len(predict_labels) == len(label_t)
            predicted_value = [1 if predict_labels[j] == label_t[j] else 0 for j in range(len(label_t))]
            predicted_value_t = torch.tensor([predicted_value], dtype=torch.float).to(self.device)

            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(action)
            
            ratio = (new_log_prob - log_prob).exp()
                    
            advantage = return_ - value
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            
            regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
            value_loss = nn.MSELoss()(return_, predicted_value_t) + regularizer
                    
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss += -self.entropy_weight * entropy.mean()
            
            loss = value_loss + policy_loss 
            self.model_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.model_optim.step()
            
            loss_list.append(loss)

                    
    
    def train(self, env, inner_update=False, loss_list = None, clip_param = 0.2):
        states, labels = env.get_init_states_and_labels()
        self.old_log_prob = None
        
        i = 0
        env_slice = len(env) // 2
        while True:
            if i * env_slice >= len(env):
                return 
            i += 1
            
            log_prob_list    = []
            value_list       = []
            reward_list      = []
            mask_list        = []
            states_list      = []
            actions_list     = []
            labels_list      = []
            batch_entropy    = 0
            
            for step in range(min(env_slice, len(env))):
                actions = self.select_action(states, labels)
                self.step(actions, env)
                
                states, value, log_prob, entropy, next_states, next_labels, reward, done = self.transition
                self.transition = list()
                
                reward = torch.tensor([reward], dtype=torch.float).to(self.device)
                batch_entropy += entropy
                
                log_prob_list.append(log_prob.detach())
                value_list.append(value.detach())
                reward_list.append(reward.detach())                            
                mask_list.append(1-done)
                states_list.append(states)
                actions_list.append(actions)
                labels_list.append(labels)
                
                states = next_states
                labels = next_labels                                
                            
                if done:
                    break
            
            # * After for loop, states is same with next_states
            next_states = states
            next_labels = labels
            predicted_value_t = 0
            
            if next_states is not None:
                _, next_critic_logit = self.model(next_states)
                _, predict_labels = torch.max(next_critic_logit.data, 1)
                assert len(predict_labels) == len(next_labels)
                predicted_value = [1 if predict_labels[j] == next_labels[j] else 0 for j in range(len(next_labels))]
                predicted_value_t = torch.tensor([predicted_value], dtype=torch.float, requires_grad=True).to(self.device)
        
            returns = self.compute_gae(predicted_value_t, reward_list, mask_list, value_list)
            
            log_prob_t = log_prob_list
            returns_t = returns
            values_t = value_list
            states_t = states_list
            actions_t = actions_list
            labels_t = labels_list
            
            if inner_update:
                self.ppo_inner_update(states_t, labels_t, actions_t, log_prob_t, returns_t, values_t, clip_param)
            else:
                self.ppo_update(states_t, labels_t, actions_t, log_prob_t, returns_t, values_t, clip_param, loss_list)
            
            
    def test(self, env):
        states, labels = env.get_init_states_and_labels()
        reward_sum = 0
        self.old_log_prob = None
        while True:
            actions = self.select_action(states, labels)
            next_states, next_labels, reward, done = self.step(actions, env)
            #self.update_model(labels)
                
            states = next_states
            labels = next_labels
            reward_sum += sum(reward)
                        
            if done:
                break
        
        return reward_sum
