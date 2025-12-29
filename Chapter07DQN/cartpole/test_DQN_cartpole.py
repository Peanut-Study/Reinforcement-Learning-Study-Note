import torch
import torch.nn.functional as F
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
'''这里把 DQN_cartpole.ipynb中的Qnet复制粘贴过来。'''
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
hidden_dim = 64
action_dim = env.action_space.n
model = Qnet(state_dim, hidden_dim, action_dim).to(device)
model.load_state_dict(torch.load('dqn_cartpole_target_q_net.pth', map_location=device))
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
