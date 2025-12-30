import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===============================
# Actor 网络定义（必须和训练时一致）。不要直接调用，因为会直接运行调用的全部代码，没写 if __name__
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
# ===============================
# 环境
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 加载训练好的 Actor
actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
actor.load_state_dict(torch.load("ppo_clip_pendulum_actor.pth", map_location=device))
actor.eval()

# ===============================
# 测试 Actor
num_episodes = 10  # 测试几个 episode
return_list = []
for ep in range(num_episodes):

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample().clamp(-2.0, 2.0).cpu().numpy()[0]

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        return_list.append(total_reward)

    print(f"Episode {ep+1} return: {total_reward}")

print("{}个episode的平均return:{}".format(num_episodes,np.mean(return_list)))
env.close()
