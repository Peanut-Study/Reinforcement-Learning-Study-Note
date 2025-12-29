import gymnasium as gym
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode="human")
# 导入策略网络模型
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# 加载模型
state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n

policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
policy.load_state_dict(torch.load("reinforce_cartpole.pth", map_location=device ))
policy.eval()

state, info = env.reset()
done = False
total_reward = 0.0

while not done:
    state_tensor = torch.from_numpy(state.reshape(1, -1)).float().to(device)

    with torch.no_grad():
        probs = policy(state_tensor)     # GPU 推理
        action = torch.distributions.Categorical(probs).sample().item()

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

print("测试总得分：", total_reward)
env.close()

