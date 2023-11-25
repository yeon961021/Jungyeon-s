#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:41:03 2023

@author: ijeong-yeon
"""
import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
import my_optim
from test import test

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    # 비동기화를 위해 rank를 통해 시드를 이동
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape(0), env.action_space)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.data
            hx = hx.data
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model(state.unsqueeze(0), (hx,cx))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, action)
            log_probs.append(log_prob)
            values.append(value)
            state, reward, done = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward,1), -1)
            if done:
                episode_length = 0
                state = env.reset
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1,1) # 누적 보상 초기화
        if not done:
            value, _, _ = model(state.unsqueeze(0), (hx,cx))
            R = value.data
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1,1) # A(a,s) = Q(a,s) - V(s)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i] 
            # R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1)*r_n-1 + gamma^nb_steps * V(last_step)
            advantage = R - values[i]
            value_loss = value_loss + (0.5 * advantage.pow(2))
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma *params.tau + TD
            policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),40) # 경사 값이 매우커지는 것을 방지
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'

# Main run
os.environ['OMP_NUM_THREADS'] = '1'
params = Params()
torch.manual_seed(params.seed)
env = create_atari_env(params.env_name)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
shared_model.share_memory()
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
optimizer.share_memory()
processes = []
p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
p.start()
processes.append(p)
for rank in range(0, params.num_processes):
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
            
            