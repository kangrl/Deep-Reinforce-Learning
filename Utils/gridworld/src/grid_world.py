__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import os
import sys
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.arguments import args

# GridWorld类：实现一个网格世界环境，用于强化学习实验
class GridWorld():

    def __init__(self, env_size=None,
                 start_state=None,
                 target_state=None,
                 forbidden_states=None):
        """
        初始化网格世界环境
        如果参数为None，则使用args中的默认值
        """
        # 初始化网格世界的基本参数
        self.env_size = env_size if env_size is not None else args.env_size  # 环境大小 (width, height)
        self.num_states = self.env_size[0] * self.env_size[1]               # 状态总数
        self.start_state = start_state if start_state is not None else args.start_state  # 起始状态
        self.target_state = target_state if target_state is not None else args.target_state  # 目标状态
        self.forbidden_states = list(forbidden_states if forbidden_states is not None else args.forbidden_states)  # 禁止状态（障碍物）

        self.agent_state = self.start_state              # 智能体当前状态
        self.action_space = list(args.action_space)      # 动作空间 - 创建副本避免共享
        self.reward_target = args.reward_target          # 到达目标的奖励
        self.reward_forbidden = args.reward_forbidden    # 碰到障碍物的惩罚
        self.reward_step = args.reward_step             # 每步的奖励

        # 可视化相关参数
        self.canvas = None
        self.animation_interval = args.animation_interval

        # 定义不同元素的颜色
        self.color_forbid = (0.9290,0.6940,0.125)      # 障碍物颜色
        self.color_target = (0.3010,0.7450,0.9330)     # 目标颜色
        self.color_policy = (0.4660,0.6740,0.1880)     # 策略箭头颜色
        self.color_trajectory = (0, 1, 0)               # 轨迹颜色
        self.color_agent = (0,0,1)                      # 智能体颜色

    # 重置环境到初始状态
    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state]
        return self.agent_state, {}

    # 执行一步动作
    def step(self, action):
        assert action in self.action_space, "Invalid action"

        # 获取下一个状态和奖励
        next_state, reward  = self.get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        # 添加一些随机噪声使轨迹显示更自然
        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        # 记录轨迹
        self.traj.append(state_store)
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}

    # 根据当前状态和动作计算下一个状态和奖励
    def get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))

        # 处理各种边界情况和特殊状态
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # 向下越界
            y = self.env_size[1] - 1
            reward = self.reward_forbidden
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # 向右越界
            x = self.env_size[0] - 1
            reward = self.reward_forbidden
        elif y - 1 < 0 and action == (0,-1):   # 向上越界
            y = 0
            reward = self.reward_forbidden
        elif x - 1 < 0 and action == (-1, 0):  # 向左越界
            x = 0
            reward = self.reward_forbidden
        elif new_state == self.target_state:  # 到达目标
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # 碰到障碍物
            x, y = state
            reward = self.reward_forbidden
        else:  # 正常移动
            x, y = new_state
            reward = self.reward_step

        return (x, y), reward

    # 判断是否到达目标状态
    def _is_done(self, state):
        return state == self.target_state

    # 可视化当前环境状态
    def render(self, animation_interval=args.animation_interval):
        if self.canvas is None:
            # 首次渲染时初始化画布
            plt.ion()
            self.canvas, self.ax = plt.subplots()
            # 设置坐标轴范围和网格
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.ax.xaxis.set_ticks_position('top')

            # 添加坐标标签
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)

            # 绘制目标状态
            self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)

            # 绘制障碍物
            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            # 创建智能体和轨迹的图形对象
            self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5)
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # 更新智能体位置和轨迹
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])
        traj_x, traj_y = zip(*self.traj)
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)
        if args.debug:
            input('press Enter to continue...')

    # 在环境中可视化策略（用箭头表示）
    def add_policy(self, policy_matrix):
        for state, state_action_group in enumerate(policy_matrix):
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):  # 非停留动作用箭头表示
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
                    else:  # 停留动作用圆圈表示
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))

    # 在网格中显示状态值
    def add_state_values(self, values, precision=1):
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='black')
