import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict


class env:
    def __init__(self, env_states_list, env_labels_list):
        self.states = env_states_list
        self.labels = env_labels_list
        self.first_states = None
        self.first_labels = None
       # print(np.array(self.states).shape)
       # print(np.array(self.labels))
       # assert 1 == 2
        
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
        
        rewards = [1 if actions[j] == current_labels[j] else 0 for j in range(len(current_labels))]
        if len(self.states) <= 1:            
            is_done = True
            next_states = None
            next_labels = None
        else:            
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
        

class A2CAgent:
    def __init__(self, model,
                 gamma, entropy_weight, 
                 class_num, device):
        self.model = model

        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.device = device
        
        self.model_fast_weight = OrderedDict(model.named_parameters())
        
        self.transition = list()
        self.predicted_reward = 0
        self.total_reward = 0
        self.inner_lr = 0.01
        
    def _reset(self):
        self.total_reward = 0
        
    def select_action(self, state):
        dist, _ = self.model(state)
        selected_action = dist.sample()
        log_prob = dist.log_prob(selected_action)
        entropy = dist.entropy()
        self.transition = [state, log_prob, entropy]
        
        return selected_action
    
    def step(self, action, env):
        next_states, next_labels, reward, done, _ = env.step(action)
        self.transition.extend([next_states, next_labels, reward, done])
        return next_states, next_labels, reward, done
    
    def update_inner_model(self, truth_labels):
        states, log_prob, entropy, next_states, next_labels, reward, done = self.transition
        self.transition = list()
        # done_mask = 1 - done
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
        
        _, critic_prob = self.model(states)
        _, predict_labels = torch.max(critic_prob.data, 1)
        
        assert len(predict_labels) == len(truth_labels)        
        predicted_value = [1 if predict_labels[j] == truth_labels[j] else 0 for j in range(len(truth_labels))]
        predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
        
        if done != True:
            # target_value = reward + self.gamma * self.critic(next_state) * done_mask
            _, next_critic_prob = self.model(next_states)
            _, next_predict_labels = torch.max(next_critic_prob.data, 1)
            assert len(next_predict_labels) == len(next_labels)
            
            predicted_next_value = [1 if next_predict_labels[j] == next_labels[j] else 0 for j in range(len(next_labels))]
            predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
            
            target_value = reward_t + self.gamma * predicted_next_value_t
        else:
            target_value = reward_t
            
#        assert len(target_value) == len(predicted_value_t), f"len(target_value) : {len(target_value)},  len(predicted_value) : {len(predicted_value)}"
        
        # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
        #regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
        value_loss = nn.MSELoss()(target_value, predicted_value_t) #+ regularizer
        
        advantage = (target_value - predicted_value_t).detach()
        policy_loss = -(log_prob * advantage)
        policy_loss += -self.entropy_weight * entropy
        
        inner_loss = value_loss + policy_loss
        
        inner_gradients = torch.autograd.grad(inner_loss.mean(), self.model_fast_weight.values(), create_graph=True, allow_unused=True)
        
        self.model_fast_weight = OrderedDict(
            (name, param - self.inner_lr * (0 if grad is None else grad))                    
            for ((name, param), grad) in zip(self.model_fast_weight.items(), inner_gradients)                    
        )   

    def update_model(self, truth_labels):
        self.model.weight = self.model_fast_weight
        
        states, log_prob, entropy, next_states, next_labels, reward, done = self.transition
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
        # done_mask = 1 - done
        
        _, critic_logit = self.model(states)
        critic_logit = critic_logit.view(-1, self.class_num)
        
        _, predict_labels = torch.max(critic_logit.data, 1)
        
        assert len(predict_labels) == len(truth_labels)
        predicted_value = [1 if predict_labels[j] == truth_labels[j] else 0 for j in range(len(truth_labels))]
        predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
        
        if next_states is not None:
            _, next_critic_logit = self.model(next_states)
            next_critic_logit = next_critic_logit.view(-1, self.class_num)
            
            _, next_predict_labels = torch.max(next_critic_logit.data, 1)
            
            assert len(next_predict_labels) == len(next_labels)
            predicted_next_value = [1 if next_predict_labels[j] == next_labels[j] else 0 for j in range(len(truth_labels))]
            predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
            
            target_value = reward_t + self.gamma * predicted_next_value_t
        else:
            target_value = reward_t
            
        #assert len(target_value) == len(predicted_value_t), f"len(target_value) : {len(target_value)},  len(predicted_value) : {len(predicted_value)}"
        #regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
        value_loss = nn.MSELoss()(target_value, predicted_value_t) #+ regularizer
        
        advantage = (target_value - predicted_value_t).detach()
        policy_loss = -(log_prob * advantage)
        policy_loss += -self.entropy_weight * entropy
        
        loss = value_loss + policy_loss
        '''        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        '''
        
        return loss

    def train(self, env, inner_update=False, loss_list=None):
        states, labels = env.get_init_states_and_labels()
        while True:
            actions = self.select_action(states)
            next_states, next_labels, reward, done = self.step(actions, env)
            
            if inner_update:
                self.update_inner_model(labels)
            else:
                loss = self.update_model(labels)
                loss_list.append(loss)
                
            states = next_states
            labels = next_labels
                        
            if done:
                break
            
    def test(self, env):
        states, labels = env.get_init_states_and_labels()
        reward_sum = 0
        while True:
            actions = self.select_action(states)
            next_states, next_labels, reward, done = self.step(actions, env)
            #self.update_model(labels)
                
            states = next_states
            labels = next_labels
            reward_sum += sum(reward)
                        
            if done:
                break
        
        return reward_sum
