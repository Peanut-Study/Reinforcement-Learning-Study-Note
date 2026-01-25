import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# ────────────────────────────────────────────────
# 1. 定义和原始训练时完全相同的 PolicyNet 类（必须一模一样）
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# ────────────────────────────────────────────────
# 2. 配置（和训练时保持一致）
state_dim = 4  # CartPole-v1 的状态维度
hidden_dim = 64
action_dim = 2  # 0: 左推, 1: 右推
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "bc_cartpole_policy_net.pth"

# ────────────────────────────────────────────────
# 3. 加载模型
policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()  # 评估模式（关闭 dropout 等，如果有的话）

print(f"已成功加载模型：{MODEL_PATH}")
print(f"使用设备：{device}")


# ────────────────────────────────────────────────
# 4. 测试函数（和训练时的 test_agent 类似）
def test_bc_agent(policy_net, env, n_episodes=10, render=False):
    """
    测试 BC 策略，跑 n_episodes 个回合，返回平均回报
    render=True 时会弹出窗口显示动画
    """
    returns = []
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            # 准备输入：加 batch 维度
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            probs = policy_net(state_tensor)  # shape: (1, action_dim)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()  # 采样动作（随机性小，因为 softmax 很尖锐）
            # action = torch.argmax(probs, dim=1).item()  # 也可以用 greedy 取最大概率动作

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state
            done = terminated or truncated

            if render:
                env.render()

        returns.append(episode_return)
        print(f"Episode {episode:3d} | Return: {episode_return:6.1f}")

    mean_return = np.mean(returns)
    print(f"\n测试完成！平均回报（{n_episodes} episodes）：{mean_return:.2f}")
    return mean_return


# ────────────────────────────────────────────────
# 5. 运行测试
env = gym.make("CartPole-v1", render_mode="human")  # 带渲染窗口

print("\n=== 开始测试 BC 策略（带渲染） ===")
test_bc_agent(policy_net, env, n_episodes=10, render=True)


env.close()