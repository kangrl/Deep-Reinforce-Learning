# 示例代码：演示如何使用GridWorld环境

import os
import sys
import time
sys.path.append("..")

from Utils.gridworld.src.grid_world import GridWorld
import random
import numpy as np

# 主程序
if __name__ == "__main__":
    # 创建网格世界环境
    env = GridWorld()
    state = env.reset()

    # 随机策略演示：智能体随机选择动作
    for t in range(1000):
        env.render()  # 可视化当前状态
        action = random.choice(env.action_space)  # 随机选择一个动作
        next_state, reward, done, info = env.step(action)  # 执行动作
        # 打印每一步的信息（状态坐标从1开始计数）
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break

    # 添加随机策略可视化
    # 为每个状态生成随机动作概率分布
    policy_matrix=np.random.rand(env.num_states, len(env.action_space))
    # 归一化，使每个状态的动作概率和为1
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]

    # 在环境中显示新策略
    env.render(animation_interval=2)  # 清除之前的策略
    env.add_policy(policy_matrix)

    # 添加随机状态值
    values = np.random.uniform(0,10,(env.num_states,))  # 生成随机状态值
    env.add_state_values(values)  # 在环境中显示状态值
    time.sleep(3)  # 暂停3秒以便观察

    # 最终渲染环境
    env.render(animation_interval=2)
