import gymnasium as gym
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import random


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def compute_advantage(gamma, lmbda, td_delta):
    advantage = 0
    advantage_list = []
    for delta in reversed(td_delta):
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.stack(advantage_list)

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs   # 一条序列的数据用于训练轮数
        self.eps = eps         # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):

        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
            td_delta = td_target - self.critic(states)
            advantage = compute_advantage(self.gamma, self.lmbda,td_delta).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions))

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))                  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            self.critic_optimizer.step()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:

            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state , _ = env.reset()
                done = False
                while done == False:
                    action = agent.take_action(state)  # take_action输出的是[action.item()]
                    next_state, reward, terminated, truncated, _ = env.step( action )
                    done = terminated or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def test_agent(env, actor, device, n_episode=10):
    actor.eval()
    return_list = []

    for ep in range(n_episode):
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
        print(f"Test Episode {ep + 1}: Return = {episode_return}")

    print("=" * 40)
    print(f"Average Return over {n_episode} episodes: {np.mean(return_list):.2f}")


if __name__ == '__main__':

    env_name = 'CartPole-v1'
    env = gym.make("CartPole-v1",)
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    actor_lr = 1e-3
    critic_lr = 3e-3
    lmbda = 0.95
    epochs = 5
    eps = 0.2
    gamma = 0.98
    device = torch.device("cuda:0")

    ppo_agent = PPO(state_dim, hidden_dim, action_dim,actor_lr, critic_lr,lmbda, epochs, eps, gamma, device)

    model_path = "ppo_agent_cartpole_actor_50.pth"

    if os.path.exists(model_path):
        print("✅ 检测到已训练模型，直接测试")
        ppo_agent.actor.load_state_dict(torch.load(model_path, map_location=device))
        test_agent(env, ppo_agent.actor, device, n_episode=10)

    else:
        print("🚀 未检测到模型，开始训练")
        num_episodes = 50
        return_list = train_on_policy_agent(env, ppo_agent, num_episodes)

        torch.save(ppo_agent.actor.state_dict(), model_path)
        print("💾 模型已保存")

        plt.figure(figsize=(8, 5))
        plt.plot(return_list)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('PPO-Clip on {}'.format(env_name))
        plt.grid(True)
        plt.show()

