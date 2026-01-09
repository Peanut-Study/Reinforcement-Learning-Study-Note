import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
env_name = 'Pendulum-v1'
env = gym.make(env_name,render_mode="human")
device = torch.device("cuda:0")
#===============================================
def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]      # 连续动作的最小值，获得环境的动作的界限
    action_upbound = env.action_space.high[0]      # 连续动作的最大值
    return action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)    # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)  # 这个状态有几个动作，优势值就有几个
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = 11   # 将连续动作分成11个离散动作
q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
q_net.load_state_dict(torch.load('dueling_dqn_pendulum_q_net.pth', map_location=device))
q_net.eval()

#===========================================================
state, info = env.reset()
done = False
total_reward = 0
while not done:
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
    # 用训练好的模型选动作
    with torch.no_grad():
        action = q_net(state_tensor).argmax().item()
    action = dis_to_con(action,env,action_dim)
    # 执行动作
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    done = terminated or truncated

    state = next_state
    total_reward += reward

print("Total reward:", total_reward)
env.close()



