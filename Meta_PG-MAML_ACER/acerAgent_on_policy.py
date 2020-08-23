import numpy as np
import torch
from torch.cuda import max_memory_allocated
import torch.nn as nn
import random

from torch.distributions import Categorical

from collections import OrderedDict
from collections import deque

from torch.nn.modules import loss


class EpisodicReplayMemory:
    def __init__(self, capacity, max_episode_length):
        # * self.num_episodes = 1000 // 20 = 50
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0
        
    def push(self, state, action, reward, policy, mask, done):
        self.buffer[self.position].append((state, action, reward, policy, mask))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)
            
    def sample(self, batch_size, max_len=None):
        min_len = 0
        rand_episodes = random.sample(self.buffer, batch_size)
        min_len = len(rand_episodes[0])
            
        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len
            
        episodes = []
        
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx+max_len])
        
        return list(map(list, zip(*episodes)))
    
    
    def __len__(self):
        return len(self.buffer)

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
        
        assert len(actions) == len(current_labels)
        #print(actions)
        #print(current_labels)
        rewards = [1 if actions[j] == current_labels[j] else 0 for j in range(len(current_labels))]
        
        if len(self.states) == 2:            
            is_done = True
        elif len(self.states) > 2:            
            is_done = False
        elif len(self.states) < 2:
            assert "At last, need 2 states"
            
        next_states = self.states[1]
        next_labels = self.labels[1]
        
        if len(self.states) >= 1:
            del self.states[0]
            del self.labels[0]

        return next_states, next_labels, rewards, is_done, current_states
        

class ACERAgent:
    def __init__(self, model, model_optim, feature_encoder_optim,
                 gamma, entropy_weight, 
                 class_num, inner_replay_buffer,
                 replay_buffer, device):
        self.model = model
        self.model_optim = model_optim
        self.feature_encoder_optim = feature_encoder_optim
        
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.device = device
        
        self.model_fast_weight = OrderedDict(model.named_parameters())
        self.inner_replay_buffer = inner_replay_buffer
        self.replay_buffer = replay_buffer
        
        self.transition = list()
        self.predicted_reward = 0
        self.total_reward = 0
        self.inner_lr = 0.01
        
    def _reset(self):
        self.total_reward = 0
          
    def select_action(self, states):
        policy, q_value, value = self.model(states)
        dist = Categorical(policy)
        selected_action = dist.sample()
        log_prob = dist.log_prob(selected_action)
        entropy = dist.entropy().mean()
        
        self.transition = [states, policy, q_value, value, log_prob, entropy]
        
        return selected_action
    
    def step(self, action, env):
        next_states, next_labels, reward, done, _ = env.step(action)
        self.transition.extend([next_states, next_labels, reward, done])
        
        return next_states, next_labels, reward, done
    
    def compute_acer_loss(self, policies, q_values,
                          values, actions, rewards,
                          retrace, masks, behavior_policies,
                          entropy_weight=None, gamma=None,
                          inner_update=False, truncation_clip=10,
                          loss_list = None):
        entropy_weight=self.entropy_weight
        gamma = self.gamma
        loss = 0
        
        for step in reversed(range(len(rewards))):
            importance_weight = policies[step].detach() / behavior_policies[step].detach()
            
            assert sum(masks[step])/len(masks[step]) != 1.0 or sum(masks[step])/len(masks[step]) != 0.0
            retrace = rewards[step].view(-1, 1) + gamma * retrace * sum(masks[step])/len(masks[step])
            advantage = retrace - values[step]
            
            log_policy_action = policies[step].gather(1, actions[step].view(-1,1)).log()
            assert log_policy_action.shape == actions[step].view(-1,1).shape, \
                f"log_policy_action.shape : {log_policy_action.shape},  actions[step].view(-1,1).shape : {actions[step].view(-1,1).shape}"
            
            truncated_importance_weight = importance_weight.gather(1, actions[step].view(-1,1)).clamp(max=truncation_clip)
            assert truncated_importance_weight.shape == actions[step].view(-1,1).shape, \
                f"truncated_importance_weight.shape : {truncated_importance_weight.shape}, actions[step].view(-1,1).shape : {actions[step].view(-1,1).shape}"
            
            actor_loss = -(truncated_importance_weight * log_policy_action * advantage.detach()).mean(0)
            correction_weight = (1 - truncation_clip / importance_weight).clamp(min=0)
            actor_loss -= (correction_weight * policies[step].log() * policies[step]).sum(1).mean(0)
            
            entropy = entropy_weight * -(policies[step].log() * policies[step]).sum(1).mean(0)

            q_value = q_values[step].gather(1, actions[step].view(-1,1))
            #critic_loss = ((retrace - q_value) ** 2 / 2).mean()
            critic_loss = nn.MSELoss()(retrace, q_value)
            
            truncated_rho = importance_weight.gather(1, actions[step].view(-1,1)).clamp(max=1)
            assert truncated_rho.shape == actions[step].view(-1,1).shape, \
                f'truncated_rho.shape : {truncated_rho.shape}, actions[step].view(-1,1).shape : {actions[step].view(-1,1).shape}'
            
            retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()
            
            loss += actor_loss + critic_loss - entropy
        
        if inner_update != True:
            loss_list.append(loss)   
        return loss

        
    def off_policy_inner_update(self, batch_size, replay_ratio=4, loss_list=None):
        if batch_size >= len(self.inner_replay_buffer):
            return

        for _ in range(replay_ratio):
            trajs = self.inner_replay_buffer.sample(batch_size)
            
            state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))
            
            q_values = []
            values = []
            policies = []
            
            for step in range(state.size(0)):
                policy, q_value, value = self.model(state[step])
                
                q_values.append(q_value)
                policies.append(policy)
                values.append(value)

            _, _, retrace = self.model(state[-1])
            
            loss = self.compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy, inner_update=True)
            inner_value_gradients = torch.autograd.grad(loss, self.model_fast_weight.values(), create_graph=True, allow_unused=True)
        
            self.model_fast_weight = OrderedDict(
                (name, param - self.inner_lr * (0 if grad is None else grad))                    
                for ((name, param), grad) in zip(self.model_fast_weight.items(), inner_value_gradients)                    
            )        
        
    def off_policy_update(self, batch_size, replay_ratio=4, loss_list=None):
        if batch_size >= len(self.replay_buffer) + 1:
            return
        
        for _ in range(np.random.poisson(replay_ratio)):
            trajs = self.inner_replay_buffer.sample(batch_size)
            state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))
            
            q_values = []
            values = []
            policies = []
            
            for step in range(state.size(0)):
                policy, q_value, value = self.model(state[step])
                
                q_values.append(q_value)
                policies.append(policy)
                values.append(value)

            _, _, retrace = self.model(state[-1])
            self.compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy, loss_list)
                
    
    def train(self, env, inner_update=False, loss_list = None):
        states, labels = env.get_init_states_and_labels()
        
        q_values = []
        values   = []
        policies = []
        actions  = []
        rewards  = []
        masks    = []
        
        for step in range(len(env)):
            action = self.select_action(states)
            self.step(action, env)
            states, policy, q_value, value, log_prob, entropy, next_states, next_labels, reward, done = self.transition
            self.transition = list()
            
            reward = torch.tensor([reward], dtype=torch.float).to(self.device)
            mask = torch.tensor([1-done], dtype=torch.float).to(self.device)
            '''
            if inner_update:
                self.inner_replay_buffer.push(states.detach(), action, reward, policy.detach(), mask, done)
            else:
                self.replay_buffer.push(states.detach(), action, reward, policy.detach(), mask, done)
            '''
            q_values.append(q_value)
            policies.append(policy)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            masks.append(mask)

            states = next_states
            labels = next_labels                                
                        
            if done:
                break
        
        # * After for loop, states is same with next_states
        next_states = states
        next_labels = labels
        
        _, _, retrace = self.model(next_states)
        retrace = retrace.detach()
        
        self.compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, policies, loss_list=loss_list, inner_update=inner_update)
        
        '''
        if inner_update:
            self.off_policy_inner_update(32)
        else:
            self.off_policy_update(32)
        '''
                    
    def test(self, env):
        env_length = len(env)
        states, labels = env.get_init_states_and_labels()
        reward_sum = 0
        reward_len = 0
        while True:
            actions = self.select_action(states)
            next_states, next_labels, reward, done = self.step(actions, env)
            #self.update_model(labels)
                
            states = next_states
            labels = next_labels
            reward_sum += sum(reward)
            reward_len += len(reward)
            if done:
                break
        
        assert reward_len != env_length
        return reward_sum
    