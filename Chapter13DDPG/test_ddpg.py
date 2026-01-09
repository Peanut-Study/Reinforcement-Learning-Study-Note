import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

env_name = 'Pendulum-v1'
env = gym.make(env_name,render_mode = 'human')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound   # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
ddpg_actor_net = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
ddpg_actor_net.load_state_dict(torch.load("ddpg_actor_net_v1.pth", map_location=device))
ddpg_actor_net.eval()

# 开始测试
num_episodes = 10# 测试几个 episode
return_list = []
for ep in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while done==False:

        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = ddpg_actor_net(state_tensor)
        action = action.detach().cpu().numpy().flatten()  # 推荐

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    return_list.append(total_reward)

    print(f"Episode {ep+1} return: {total_reward}")

print("{}个episode的平均return:{}".format(num_episodes,np.mean(return_list)))
env.close()
