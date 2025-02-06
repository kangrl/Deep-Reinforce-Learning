"""
展示不同类型的策略可视化示例
"""
import os
import random
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.grid_world import GridWorld

def show_policy_example():
    # 创建环境
    env = GridWorld()
    state = env.reset()
    env.render()

    # # 示例1：确定性策略 - 所有状态都向右移动
    # print("\n示例1：确定性策略 - 所有状态都向右移动")
    # policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    # right_action_idx = env.action_space.index((1, 0))  # 找到向右移动(1,0)的索引
    # policy_matrix[:, right_action_idx] = 1.0  # 所有状态都设置为向右移动
    # env.add_policy(policy_matrix)
    # env.render(animation_interval=2)
    # time.sleep(2)  # 暂停2秒观察

    # # 示例2：概率性策略 - 向右和向下的混合策略
    # print("\n示例2：概率性策略 - 向右和向下的混合策略")
    # policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    # right_idx = env.action_space.index((1, 0))   # 向右的动作索引
    # down_idx = env.action_space.index((0, 1))    # 向下的动作索引
    # policy_matrix[:, right_idx] = 0.7  # 70%的概率向右移动
    # policy_matrix[:, down_idx] = 0.3   # 30%的概率向下移动
    # env.render(animation_interval=2)  # 清除之前的策略
    # env.add_policy(policy_matrix)
    # time.sleep(2)  # 暂停2秒观察

    # 示例3：每个状态都有不同的策略
    print("\n示例3：每个状态都有不同的策略")
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for state in range(env.num_states):
        x = state % env.env_size[0]  # 获取状态的x坐标
        y = state // env.env_size[0]  # 获取状态的y坐标

        if x < env.env_size[0] - 1:  # 如果不在最右边，主要往右走
            policy_matrix[state, env.action_space.index((1, 0))] = 0.8  # 80%向右
            if y < env.env_size[1] - 1:  # 如果不在最下边，偶尔往下走
                policy_matrix[state, env.action_space.index((0, 1))] = 0.2  # 20%向下
        else:  # 在最右边时
            if y < env.env_size[1] - 1:  # 如果不在最下边，就往下走
                policy_matrix[state, env.action_space.index((0, 1))] = 1.0  # 100%向下
            else:  # 在右下角时停留
                policy_matrix[state, env.action_space.index((0, 0))] = 1.0  # 100%停留

    env.render(animation_interval=2)  # 清除之前的策略
    env.add_policy(policy_matrix)
    time.sleep(200)  # 暂停2秒观察

    # # 示例4：每个状态都是随机策略
    # print("\n示例4：每个状态都是随机策略")
    # policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    # for i in range(env.num_states):  # 遍历所有可能的动作
    #     probs = np.random.dirichlet([1]*len(env.action_space))# 随机选择一个动作
    #     indx = np.argmax(probs)
    #     policy_matrix[i, indx] = 1.  # 使用Dirichlet分布生成概率，使每个状态的概率和为1

    # env.render(animation_interval=2)  # 清除之前的策略
    # env.add_policy(policy_matrix)
    # time.sleep(2000)  # 暂停2秒观察

if __name__ == "__main__":
    show_policy_example()
