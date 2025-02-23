'''
Author: kangrl
Email: kangrl@live.cn
Date: 2025-02-22 21:34:44
LastEditors: kangrl
LastEditTime: 2025-02-22 22:17:48
FilePath: /Deep-Reinforce-Learning/Utils/tools.py
Copyright (C) 2025 by kangrl, All Rights Reserved.
Description:
'''
import random

import torch
import numpy as np
import collections


class ReplayBuffer:
    """ Experience Replay Buffer """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)


    def add(self, state, action, reward, next_state, done):
        """ Add Experience to Buffer """

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """ Sample a Batch of Experiences from Buffer """

        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """ Return the Size of Buffer """

        return len(self.buffer)


def get_device():
    """ Get device """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
