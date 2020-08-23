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
        
        del self.states[0]
        del self.labels[0]
        
        return self.first_states, self.first_labels
        
    def step(self, actions):
        rewards = 0
        is_done = None
        current_states = None
        current_labels = None
        
        next_states = None
        next_labels = None
        
        if len(self.states) <= 1:
            if len(self.states) == 1:
                current_states = self.states[0]
                current_labels = self.labels[0]
            else:
                current_states = self.first_states
                current_labels = self.first_labels
            
            rewards = [1 if actions[j] == current_labels[j] else 0 for j in range(len(current_labels))]
            is_done = True
            # * To understand code easily duplicated it 
            next_states = None
            next_labels = None
        else:
            current_states = self.states[0]
            current_labels = self.labels[0]
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
        

class A2CAgent:
    def __init__(self, actor, critic,
                 gamma, entropy_weight, 
                 input_size, hidden_size, 
                 class_num, device):
        self.actor = actor
        self.critic = critic
        
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.device = device
        
        self.actor_fast_weight = OrderedDict(actor.named_parameters())
        self.critic_fast_weight = OrderedDict(critic.named_parameters())
        
        self.transition = list()
        self.predicted_reward = 0
        self.total_reward = 0
        self.inner_lr = 0.01
        
    def _reset(self):
        self.total_reward = 0
        
    def select_action(self, state):
        dist = self.actor(state)
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
        
        critic_logit = self.critic(states).view(-1, self.class_num)
        _, predict_labels = torch.max(critic_logit.data, 1)
        predicted_value = [1 if predict_labels[j] == truth_labels[j] else 0 for j in range(len(truth_labels))]
        predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
        
        if next_states is not None:
            # target_value = reward + self.gamma * self.critic(next_state) * done_mask
            next_critic_logit = self.critic(next_states).view(-1, self.class_num)
            _, next_predict_labels = torch.max(next_critic_logit.data, 1)
            predicted_next_value = [1 if next_predict_labels[j] == next_labels[j] else 0 for j in range(len(next_labels))]
            predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
            
            target_value = reward_t + self.gamma * predicted_next_value_t
        else:
            target_value = reward_t
            
#        assert len(target_value) == len(predicted_value_t), f"len(target_value) : {len(target_value)},  len(predicted_value) : {len(predicted_value)}"
        
        # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
        regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
        value_loss = nn.MSELoss()(predicted_value_t, target_value) + regularizer.mean()
        
        advantage = (target_value - predicted_value_t).detach()
        policy_loss = -(log_prob * advantage)
        policy_loss += -self.entropy_weight * entropy
        
        inner_value_gradients = torch.autograd.grad(value_loss, self.critic_fast_weight.values(), create_graph=True, allow_unused=True)
        inner_actor_gradients = torch.autograd.grad(policy_loss.mean(), self.actor_fast_weight.values(), create_graph=True, allow_unused=True)
        
        self.actor_fast_weight = OrderedDict(
            (name, param - self.inner_lr * (0 if grad is None else grad))                    
            for ((name, param), grad) in zip(self.critic_fast_weight.items(), inner_value_gradients)                    
        )   
        
        self.actor_fast_weight = OrderedDict(
            (name, param - self.inner_lr * (0 if grad is None else grad))                    
            for ((name, param), grad) in zip(self.actor_fast_weight.items(), inner_actor_gradients)                    
        )             
        
    def update_model(self, truth_labels):
        self.actor.weight = self.actor_fast_weight
        self.critic.weight = self.critic_fast_weight
        
        states, log_prob, entropy, next_states, next_labels, reward, done = self.transition
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
        # done_mask = 1 - done
        
        critic_logit = self.critic(states).view(-1, self.class_num)
        _, predict_labels = torch.max(critic_logit.data, 1)
        predicted_value = [1 if predict_labels[j] == truth_labels[j] else 0 for j in range(len(truth_labels))]
        predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
        
        if next_states is not None:
            next_critic_logit = self.critic(next_states).view(-1, self.class_num)
            _, next_predict_labels = torch.max(next_critic_logit.data, 1)
            predicted_next_value = [1 if next_predict_labels[j] == next_labels[j] else 0 for j in range(len(truth_labels))]
            predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
            
            target_value = reward_t + self.gamma * predicted_next_value_t
        else:
            target_value = reward_t
            
#        assert len(target_value) == len(predicted_value_t), f"len(target_value) : {len(target_value)},  len(predicted_value) : {len(predicted_value)}"
        regularizer = critic_logit.pow(2).sum() / (2.0 * critic_logit.size(0))
        value_loss = nn.MSELoss()(predicted_value_t, target_value) + regularizer
        
        advantage = (target_value - predicted_value_t).detach()
        policy_loss = -(log_prob * advantage)
        policy_loss += -self.entropy_weight * entropy
        
        '''        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        '''
        
        return policy_loss, value_loss

    def train(self, env, inner_update=False, policy_loss_list=None, value_loss_list=None):
        states, labels = env.get_init_states_and_labels()
        while True:
            actions = self.select_action(states)
            next_states, next_labels, reward, done = self.step(actions, env)
            
            if inner_update:
                self.update_inner_model(labels)
            else:
                policy_loss, value_loss = self.update_model(labels)
                policy_loss_list.append(policy_loss)
                value_loss_list.append(value_loss)
                
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
