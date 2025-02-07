'''
Author: kangrl
Email: kangrl@live.cn
Date: 2025-02-07 21:31:45
LastEditors: kangrl
LastEditTime: 2025-02-07 22:35:32
FilePath: /Deep-Reinforce-Learning/03 Algorithms/3.1 Value Policy.py
Copyright (C) 2025 by kangrl, All Rights Reserved.
Description:
'''

import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from networkx import bidirectional_dijkstra
import numpy as np

from Utils.gridworld.examples.arguments import args
from Utils.gridworld.src.grid_world import GridWorld


class ValueIteration:
    """ Value Iteration for Grid World """

    def __init__(self, env, gamma=0.95, delta=0.01):
        self.env = env
        self.gamma = gamma
        self.delta = delta

        self.state_values = np.zeros(env.num_states)  # Value function
        self.policy = np.zeros((env.num_states, len(env.action_space)))  # Policy

    def _get_position(self, number):
        x = number % self.env.env_size[0]
        y = number // self.env.env_size[0]

        return x, y

    def _get_number(self, state):

        return state[0] * self.env.env_size[0] + state[1]

    def value_iteration(self):
        while True:
            error = 0
            state_values = np.zeros(env.num_states)
            for state_number, state_action_group in enumerate(self.policy):
                self.env.agent_state = self._get_position(state_number)
                action_values = []
                for i, action_probability in enumerate(state_action_group):
                    next_state, reward, done, info = self.env.step(self.env.action_space[i])
                    action_value = reward + self.gamma * self.state_values[self._get_number(next_state)] * (1 - done)
                    action_values.append(action_value)
                best_action_value = max(action_values)
                state_values[state_number] = best_action_value
                self.policy[state_number] = [1 / action_values.count(best_action_value) if action_value == best_action_value else 0 for action_value in action_values]
                error = max(abs(self.state_values[state_number] - state_values[state_number]), error)

            self.state_values = state_values
            if error < self.delta: break


if __name__ == "__main__":

    env = GridWorld()
    env.reset()
    env.render()

    agent = ValueIteration(env, gamma=0.9, delta=1e-5)
    agent.value_iteration()
    print(f'Final Policy: {agent.policy}')
    print(f'State Values: {agent.state_values}')

    env.reset()
    env.add_policy(agent.policy)
    env.add_state_values(agent.state_values)
    env.render(animation_interval=2)
    time.sleep(300)
