'''
Author: kangrl
Email: kangrl@live.cn
Date: 2025-02-18 09:27:17
LastEditors: kangrl
LastEditTime: 2025-02-18 09:30:58
FilePath: \Deep-Reinforce-Learning\gym_test.py
Copyright (C) 2025 by kangrl, All Rights Reserved.
Description:
'''

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")

for _ in range(100):
    observation, info = env.reset()
    print('observation:', observation)
