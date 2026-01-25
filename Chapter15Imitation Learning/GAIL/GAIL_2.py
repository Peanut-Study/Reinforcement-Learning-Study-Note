import gymnasium as gym
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import random

from GAIL_1 import PolicyNet , PPO

def sample_expert_data(actor, env, n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state, _ = env.reset()
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = actor(state_t)
            action = torch.distributions.Categorical(probs).sample().item()
            states.append(state)
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
    return np.array(states), np.array(actions)


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d,device):
        self.device=device
        self.discriminator = Discriminator(state_dim, hidden_dim,action_dim).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):

        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a,dtype=torch.long).to(self.device)

        agent_states = torch.tensor(np.array(agent_s), dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a,dtype=torch.long).to(self.device)

        expert_actions = F.one_hot(expert_actions, num_classes=action_dim).float()
        agent_actions = F.one_hot(agent_actions, num_classes=action_dim).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        loss_fn = nn.BCELoss()
        discriminator_loss = (  loss_fn(expert_prob, torch.ones_like(expert_prob))  +
                                loss_fn(agent_prob, torch.zeros_like(agent_prob))      )


        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(1 - agent_prob + 1e-8)
        rewards = rewards.detach().cpu().numpy().flatten()

        transition_dict = {     'states': agent_s,
                                'actions': agent_a,
                                'rewards': rewards,
                                'next_states': next_s,
                                'dones': dones            }
        self.agent.update(transition_dict)

#===============================================
if __name__=='__main__':
    env_name = 'CartPole-v1'
    env = gym.make("CartPole-v1")
    device = torch.device('cuda:0')
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    ppo_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    ppo_actor.load_state_dict(torch.load("ppo_agent_cartpole_actor_50.pth", map_location=device))
    ppo_actor.eval()
    # ====================================================================
    expert_episode = 5
    expert_s, expert_a = sample_expert_data(ppo_actor, env, expert_episode)
    print("采集专家数据完成，样本数:", expert_s.shape[0])
    n_samples = 100
    random_index = random.sample(range(expert_s.shape[0]), n_samples)  # 随机挑选 n_samples 个不重复的索引
    expert_s = expert_s[random_index]  # (100, 4)
    expert_a = expert_a[random_index]  # (100,)
    print(expert_s.shape)
    print(expert_a.shape)
    print("随机挑选 100 条样本完成")
    #====================================================================
    actor_lr = 1e-3
    critic_lr = 1e-2
    lmbda = 0.95
    gamma = 0.98
    epochs = 10
    eps = 0.2
    ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma, device)
    #===============================================
    lr_d = 1e-3
    gail = GAIL(ppo_agent, state_dim, action_dim, hidden_dim, lr_d,device)
    n_episode = 200
    return_list = []

    with tqdm(total=n_episode, desc="进度条") as pbar:
        for i in range(n_episode):
            episode_return = 0
            state , _= env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = ppo_agent.take_action(state)
                next_state, reward, terminated,truncated, _ = env.step(action)
                done = terminated or truncated

                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)

                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,next_state_list, done_list)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    print('训练完成')
    # ⭐️ ⭐️ ⭐️ ⭐️ ⭐️ ⭐️ ⭐️ ⭐️ ⭐️ ⭐️
    torch.save(ppo_agent.actor.state_dict(), "gail_cartpole_actor.pth")
    # GAIL 里又新训练了一个 PPO，只是结构一样，不是同一个网络
    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on {}'.format(env_name))
    plt.show()