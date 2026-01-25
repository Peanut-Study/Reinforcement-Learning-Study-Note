import gymnasium as gym
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import random

from GAIL_1 import PolicyNet
env_name = 'CartPole-v1'
env = gym.make("CartPole-v1")
device = torch.device('cuda:0')
state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
actor.load_state_dict(torch.load("gail_cartpole_actor.pth", map_location=device))
actor.eval()

n_test_episode = 10
return_list = []
for ep in range(n_test_episode):
    state, _ = env.reset()
    done = False
    episode_return = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = actor(state_t)
        action = torch.argmax(probs, dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

    return_list.append(episode_return)
    print(f"Episode {ep + 1}: Return = {episode_return}")

print("=" * 40)
print(f"Average Return over {n_test_episode} episodes: {np.mean(return_list):.2f}")








