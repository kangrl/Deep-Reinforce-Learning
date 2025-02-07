__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse

def str_to_tuple(s):
    """Convert string to tuple of integers"""
    try:
        return tuple(map(int, s.split(',')))
    except:
        return None

def str_to_tuple_list(s):
    """Convert string to list of tuples"""
    try:
        return [tuple(map(int, pair.split(','))) for pair in s.split(';')]
    except:
        return None

# 默认参数
default_params = {
    'env_size': (5, 5),
    'start_state': (0, 0),
    'target_state': (2, 3),
    'forbidden_states': [(1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)],
    'reward_target': 10.0,
    'reward_forbidden': -5.0,
    'reward_step': -1.0,
    'action_space': [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)],  # down, right, up, left, stay
    'debug': False,
    'animation_interval': 0.2
}

# 创建一个简单的命名空间对象来存储参数
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# 如果在命令行中运行，则使用argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grid World Environment")

    # User settings
    parser.add_argument("--env-size", type=str_to_tuple, default="5,5")
    parser.add_argument("--start-state", type=str_to_tuple, default="0,0")
    parser.add_argument("--target-state", type=str_to_tuple, default="2,3")
    parser.add_argument("--forbidden-states", type=str_to_tuple_list, default="1,1;2,1;2,2;1,3;3,3;1,4")
    parser.add_argument("--reward-target", type=float, default=10.0)
    parser.add_argument("--reward-forbidden", type=float, default=-5.0)
    parser.add_argument("--reward-step", type=float, default=-1.0)

    # Advanced settings
    parser.add_argument("--action-space", type=str_to_tuple_list, default="0,1;1,0;0,-1;-1,0;0,0")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--animation-interval", type=float, default=0.2)

    args = parser.parse_args()
else:
    # 如果在其他环境（如Jupyter）中导入，直接使用默认值
    args = Args(**default_params)
