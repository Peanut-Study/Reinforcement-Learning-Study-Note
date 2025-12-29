import torch
import torch.nn.functional as F
import torch.nn as nn
import gymnasium as gym
#========================================================================
env_name = 'CartPole-v1'
env = gym.make(env_name,render_mode="human")
#=========================================================================
try:
    device = torch.device("cuda:0")
    torch.zeros(1).to(device)
    print(f"成功使用 GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    raise RuntimeError(f"无法正常使用 GPU 0: {e}.")
#=========================================================================
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = QNet(state_dim, action_dim).to(device)
model.load_state_dict(torch.load('SarsaNN_cartpole_q_net.pth', map_location=device))
model.eval()
#================================================================================
state, info = env.reset()
done = False
total_reward = 0
while not done:
    # 转 tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    # 用训练好的模型选动作
    with torch.no_grad():
        action = model(state_tensor).argmax().item()
    # 执行动作
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    state = next_state
    total_reward += reward

print("Total reward:", total_reward)
env.close()
