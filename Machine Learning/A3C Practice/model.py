#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:16:53 2023

@author: ijeong-yeon
"""
# [1] Creating the Model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size()) # 정규분포를 따르는 랜덤 가중치를 가지는 토치 센서 초기화
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

# 가중치 초기화
def weights_init(m):
    classname = m.__.class__.__name__
    if classname.find('Conv') != -1: # 합성곱이 있는 경우
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) # 1, 2, 3 차원
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0] # 0, 2, 3 차원
        w_bound = np.sqrt( 6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound,  w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Liner') != -1: # FC가 있는 경우
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt( 6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound,  w_bound)
        m.bias.data.fill_(0)     
        
# A3C Brain
class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.lstm = nn.LSTMCell(32*3*3, 256)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear_weight.data, std = 0.01)
        self.actor_linear.bias.data.fill(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.actor_linear_weight.data, std = 1)
        self.critic_linear.bias.data.fill(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()
    def forawrd(self, inputs):
        inputs,(hx, cx) = inputs
        x = F.elu(self.conv1(inputs)) # F.elu = 더 정교해진 ReLU 함수 
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*3*3)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
