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

env_name = 'CartPole-v1'
env = gym.make("CartPole-v1")
model = PPO.load('ppo_cartpole_experts.zip', env=env, device="cpu")

def sample_expert_data(model, env, n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            states.append(state)
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
    return np.array(states), np.array(actions)

n_episode = 5
expert_s, expert_a = sample_expert_data(model, env, n_episode)
print("采集专家数据完成，样本数:", expert_s.shape[0])
n_samples = 100
random_index = random.sample(range(expert_s.shape[0]), n_samples)  # 随机挑选 n_samples 个不重复的索引
expert_s = expert_s[random_index]  # (100, 4)
expert_a = expert_a[random_index]  # (100,)
print(expert_s.shape)
print(expert_a.shape)
print("随机挑选 100 条样本完成")


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.device = device
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).view(-1, 1).to(device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs)  # 最大似然估计

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)


state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_dim = env.action_space.n
lr = 1e-3
device = torch.device("cuda:0")
bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)

n_iterations = 1000
batch_size = 64
test_returns = []

with tqdm(total=n_iterations, desc="进度条") as pbar:
    for i in range(n_iterations):
        sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=batch_size)
        bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
        current_return = test_agent(bc_agent, env, 5)
        test_returns.append(current_return)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
        pbar.update(1)
print('训练完成，保存网络参数及绘图')
torch.save(bc_agent.policy.state_dict(), 'bc_cartpole_policy_net.pth')

iteration_list = list(range(len(test_returns)))
plt.plot(iteration_list, test_returns)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('BC on {}'.format(env_name))
plt.show()