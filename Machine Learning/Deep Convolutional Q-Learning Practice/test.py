#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 3 10:39:03 2023

@author: ijeong-yeon
"""

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing
import numpy as np
from collections import namedtuple, deque
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Part 1 = Building the AI

# Making the brain
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()    # 흑백 이미지 = 1, out_channels = 특징 탐지기 개수
        self.convolutiona1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolutiona2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolutiona3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons(1, 80, 80) , out_features = 40)
        self.fc2 = nn.Linear(in_features = 40 , out_features = number_actions)

    def count_neurons(self, image_dim):
        x = torch.rand(1, *image_dim) #fake image creating
        x = F.relu(F.max_pool2d(self.convolutiona1(x), kernel_size = 3, strides = 2))
        x = F.relu(F.max_pool2d(self.convolutiona2(x), kernel_size = 3, strides = 2))
        x = F.relu(F.max_pool2d(self.convolutiona3(x), kernel_size = 3, strides = 2))
        return x.data.view(1, -1).size(1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolutiona1(x), kernel_size = 3, strides = 2))
        x = F.relu(F.max_pool2d(self.convolutiona2(x), kernel_size = 3, strides = 2))
        x = F.relu(F.max_pool2d(self.convolutiona3(x), kernel_size = 3, strides = 2))
        x = x.view(x.size(0), -1) #flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Making the body
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
        
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
# Making the AI
class AI():
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    def __call__(self, inputs):
        input = torch.from_numpy(np.array(inputs, dtype = np.float32))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()
    
# Making the AI progress on several (n_step) steps

class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
    
    def __iter__(self):
        state = self.env.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()    
# Part 2 = Implemeting Deep Convolutional Q-Learning'

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T=1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = NStepProgress(doom_env, ai, 10)
memory = ReplayMemory(n_steps = n_steps, capacity = 10000)

# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for i in batch:
        input = torch.from_numpy(np.array([i[0].state, i[-1].state], dtype = np.float32)) # 처음 상태와 마지막 입력 상태
        output = cnn(input) # Q값은 index 1에 저장
        cumul_reward = 0.0 if i[-1].done else output[1].data.max
        for step in reversed(i[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = i[0].state
        target = output[0].data
        target[i[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)
# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_reward[0]
    def average(self):
        return np.mean(self.list_of_reward)
ma = MA(100)
# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
n_epochs = 100
for i in range(1, n_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128): #32/64/128 (n단계의 크기에 따라 다르게 고름)
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_step = n_steps.rewards_steps()
    ma.add(rewards_step)
    avg_reward = ma.average()
    print('Epoch: %s, Average Reward: %s' % (str(i), str(avg_reward)))
    