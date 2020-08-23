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
       
    def __len__(self):
        return len(self.states)
        
    def get_init_states_and_labels(self):
        self.first_states = self.states[0]
        self.first_labels = self.labels[0]
        
        return self.first_states, self.first_labels
        
    def step(self, actions, last_state=False):        
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
        self.inner_lr = 0.01
        
    def select_action(self, state, model):
        dist, critic_prob = model(state)
        selected_action = dist.sample()
        log_prob = dist.log_prob(selected_action)
        entropy = dist.entropy()
        self.transition = [state, log_prob, entropy]
        
        return selected_action, critic_prob
    
    def step(self, action, env, last_state=False):
        next_states, next_labels, reward, done, _ = env.step(action, last_state)
        self.transition.extend([next_states, next_labels, reward, done])
        return next_states, next_labels, reward, done
    
    def compute_returns(self, next_values, rewards, masks):
        R = next_values
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)

        return returns
    
    def train(self, env, model):
        #next_states_list = []
        reward_list = []
        value_list = []
        log_prob_list = []
        mask_list = []
        entropy_mean_sum = 0
        
        prediction_loss = []
        
        states, labels = env.get_init_states_and_labels()
        next_states = None
            
        while True:
            actions, critic_prob = self.select_action(states, model)
            _, predict_labels = torch.max(critic_prob.data, 1)
            prediction_loss.append(nn.CrossEntropyLoss()(critic_prob, labels))
            
            assert len(predict_labels) == len(labels)
            predicted_value = [1 if predict_labels[j] == labels[j] else 0 for j in range(len(labels))]
            predicted_value_t = torch.tensor(predicted_value, dtype=torch.float).to(self.device)
            
            next_states, next_labels, reward, done = self.step(actions, env)
            if next_states == None:
                break
            
            _, log_prob, entropy, _, _, _, _ = self.transition
            self.transition = list()
            
            entropy_mean_sum += entropy.mean()
            #next_states_list.append(next_states)
            reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
            reward_list.append(reward_t)
            value_list.append(predicted_value_t)
            log_prob_list.append(log_prob)
            mask_list.append(1-done)

            states = next_states
            labels = next_labels

        next_states = states            
        next_labels = labels
        _, next_critic_prob = model(next_states)
        _, predict_labels = torch.max(next_critic_prob.data, 1)
        assert len(predict_labels) == len(next_labels)
        predicted_next_value = [1 if predict_labels[j] == next_labels[j] else 0 for j in range(len(next_labels))]
        predicted_next_value_t = torch.tensor(predicted_next_value, dtype=torch.float).to(self.device)
        returns = self.compute_returns(predicted_next_value_t, reward_list, mask_list)
        
        log_prob_t = torch.cat(log_prob_list)
        returns_t = torch.cat(returns)
        values_t = torch.cat(value_list)
        
        assert returns_t.shape == values_t.shape, f"returns_t.shape : {returns_t.shape}, values_t.shape : {values_t.shape}"
        advantage = returns_t - values_t
        actor_loss = -(log_prob_t * advantage.detach()).mean() - self.entropy_weight * entropy_mean_sum
        critic_loss = nn.MSELoss()(returns_t, values_t) * 0.5 + torch.stack(prediction_loss).mean()
        
        loss = actor_loss + critic_loss
        
        return loss
            
        
            
    def test(self, env, model):
        states, labels = env.get_init_states_and_labels()
        reward_sum = 0
        while True:
            actions, _ = self.select_action(states, model)
            next_states, next_labels, reward, done = self.step(actions, env)
            #self.update_model(labels)
                
            states = next_states
            labels = next_labels
            reward_sum += sum(reward)
                        
            if done:
                break
        
        return reward_sum
