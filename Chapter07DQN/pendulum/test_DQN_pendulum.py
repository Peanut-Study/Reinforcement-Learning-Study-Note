import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
#========================================================================
env_name = 'Pendulum-v1'
env = gym.make(env_name,render_mode="human")
#=========================================================================
try:
    device = torch.device("cuda:0")
    torch.zeros(1).to(device)
    print(f"成功使用 GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    raise RuntimeError(f"无法正常使用 GPU 0: {e}.")
#=========================================================================
class DiscreteToContinuous:
    def __init__(self, env, step_size):
        self.low, self.high = env.action_space.low[0], env.action_space.high[0]
        if not (0 < step_size <= self.high - self.low):
            raise ValueError("step_size must be in (0, high - low]")

        vals = self.low + np.arange(int(np.floor((self.high - self.low) / step_size)) + 1) * step_size
        if vals[-1] < self.high:
            vals = np.append(vals, self.high)

        self.values = vals.astype(np.float32)
        self.action_dim = len(self.values)

    def get_discrete_action_dim(self):
        return self.action_dim

    def to_continuous(self, discrete_action):
        if not 0 <= discrete_action < self.action_dim:
            raise IndexError("discrete_action out of range")
        return np.array([self.values[discrete_action]], dtype=np.float32)
step_size = 0.2
discrete_to_continuous = DiscreteToContinuous(env,step_size)
action_dim = discrete_to_continuous.get_discrete_action_dim()

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
model = Qnet(state_dim, hidden_dim, action_dim).to(device)
model.load_state_dict(torch.load('dqn_pendulum_q_net.pth', map_location=device))
model.eval()
#================================================================
num_episodes = 1

for ep in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_return = 0.0

    while not done:
        # 1. state → tensor（显式 batch 维度）
        state_tensor = (torch.from_numpy(state).float().unsqueeze(0).to(device))
        # 2. DQN 选动作（无探索，纯贪婪）
        with torch.no_grad():
            action_discrete = model(state_tensor).argmax(dim=1).item()
        # 3. 离散 → 连续
        action_continuous = discrete_to_continuous.to_continuous(action_discrete)
        # 4. 与环境交互
        next_state, reward, terminated, truncated, info = env.step(action_continuous)
        done = terminated or truncated

        episode_return += reward
        state = next_state

    print(f"Episode {ep + 1}, Return: {episode_return:.2f}")

env.close()
