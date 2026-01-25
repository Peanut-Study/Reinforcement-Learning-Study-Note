'''
生成专家策略
'''
import gymnasium as gym
from stable_baselines3 import PPO
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import random

# ========================
# 配置部分（方便修改）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo_cartpole_experts.zip")  # 保存/加载的文件路径
TOTAL_TIMESTEPS = 10000  # 如果要重新训练，用这个步数
RETRAIN = False  # 设为 True 强制重新训练，否则优先加载已有模型

# ========================
# 1. 创建环境（带渲染），训练时也会有动画
env_name = "CartPole-v1"
env = gym.make("CartPole-v1", render_mode="human")

# ========================
# 2. 加载或训练模型
if os.path.exists(MODEL_PATH) and not RETRAIN:
    print(f"检测到已有模型 {MODEL_PATH}，直接加载,然后测试...")
    model = PPO.load(MODEL_PATH, env=env, device="cpu")  # 加载时传入 env，确保兼容

else:
    print(f"没有找到模型或强制重新训练，开始从头训练...")

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # verbose=0,不显示训练记录，1，显示，2，详细显示
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    # 训练完立即保存
    model.save(MODEL_PATH)
    print(f"训练完成，已保存模型到: {MODEL_PATH}")

# ========================
# 3. 测试阶段：玩几个 episode 并打印每个回合的总奖励,看看专家策略的效果怎么样
print("\n=== 开始测试，已训练模型的表现 ===")
experts_return_list = []
for episode in range(0, 10):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        env.render()  # 实时渲染
    experts_return_list.append(episode_reward)
    print(f"Episode {episode:3d} | Total Reward: {episode_reward:6.1f}")
print(' 测试结束,专家策略的平均回报:',sum(experts_return_list)/len(experts_return_list))  # 351.9
env.close()           # 必须关闭渲染窗口