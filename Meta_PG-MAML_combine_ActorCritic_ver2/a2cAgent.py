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
    def __init__(self, gamma, entropy_weight, 
                 class_num, device):
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.device = device
        
        self.transition = list()
        self.predicted_reward = 0
        self.total_reward = 0
        self.inner_lr = 0.01
        
    def _reset(self):
        self.total_reward = 0
        
    def select_action(self, model, state):
        dist, _ = model(state)
        selected_action = dist.sample()
        log_prob = dist.log_prob(selected_action)
        entropy = dist.entropy()
        self.transition = [state, log_prob, entropy]
        
        return selected_action
    
    def step(self, action, env):
        next_states, next_labels, reward, done, _ = env.step(action)
        self.transition.extend([next_states, next_labels, reward, done])
        return next_states, next_labels, reward, done
    
    def update_model(self, model, truth_labels):
        states, log_prob, entropy, next_states, next_labels, reward, done = self.transition
        self.transition = list()
        # done_mask = 1 - done
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
        
        _, critic_prob = model(states)
        _, predict_labels = torch.max(critic_prob.data, 1)
        
        assert len(predict_labels) == len(truth_labels)        
        predicted_value = [1 if predict_labels[j] == truth_labels[j] else 0 for j in range(len(truth_labels))]
        predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
        
        if done != True:
            # target_value = reward + self.gamma * self.critic(next_state) * done_mask
            _, next_critic_prob = model(next_states)
            _, next_predict_labels = torch.max(next_critic_prob.data, 1)
            assert len(next_predict_labels) == len(next_labels)
            
            predicted_next_value = [1 if next_predict_labels[j] == next_labels[j] else 0 for j in range(len(next_labels))]
            predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
            
            target_value = reward_t + self.gamma * predicted_next_value_t
        else:
            target_value = reward_t
            
#        assert len(target_value) == len(predicted_value_t), f"len(target_value) : {len(target_value)},  len(predicted_value) : {len(predicted_value)}"
        
        # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
        regularizer = critic_prob.pow(2).sum() / (2.0 * critic_prob.size(0))
        value_loss = nn.MSELoss()(target_value, predicted_value_t) + regularizer
        
        advantage = (target_value - predicted_value_t).detach()
        policy_loss = -(log_prob * advantage)
        policy_loss += -self.entropy_weight * entropy
        
        loss = value_loss + policy_loss
        
        return loss
        
          

    def train(self, env, model, loss_list=None):
        states, labels = env.get_init_states_and_labels()
        while True:
            actions = self.select_action(model, states)
            next_states, next_labels, reward, done = self.step(actions, env)
            
            loss = self.update_model(model, labels)
            loss_list.append(loss)
                
            states = next_states
            labels = next_labels
                        
            if done:
                break
            
    def test(self, env, model):
        states, labels = env.get_init_states_and_labels()
        reward_sum = 0
        while True:
            actions = self.select_action(model, states)
            next_states, next_labels, reward, done = self.step(actions, env)
            #self.update_model(labels)
                
            states = next_states
            labels = next_labels
            reward_sum += sum(reward)
                        
            if done:
                break
        
        return reward_sum
