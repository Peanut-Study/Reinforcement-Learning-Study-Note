import numpy as np
import torch
import gymnasium as gym

from mpc import EnsembleDynamicsModel, FakeEnv, CEM

device = torch.device("cuda:0" )

# ------------------ 配置环境 ------------------
env_name = "Pendulum-v1"
env = gym.make(env_name,)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# ------------------ 加载训练好的模型 ------------------
ensemble_model = EnsembleDynamicsModel(obs_dim, action_dim)
ensemble_model.model.load_state_dict(torch.load("pets_dynamics_model_10.pth", map_location=device))
ensemble_model.model.to(device)
ensemble_model.model.eval()  # 切换到评估模式
for param in ensemble_model.model.parameters():
    param.requires_grad = False
# ------------------ 构造 FakeEnv 和 CEM ------------------
fake_env = FakeEnv(ensemble_model)
n_sequence = 50      # 生成多少条动作序列
elite_ratio = 0.2    # CEM精英比例
plan_horizon = 25    # MPC规划步长

cem = CEM(n_sequence, elite_ratio, fake_env, upper_bound, lower_bound)

# ------------------ MPC 测试 ------------------
def run_mpc_episode(env, cem, plan_horizon, action_dim):
    obs, _ = env.reset()
    done = False
    total_return = 0.0

    # 初始化动作均值和方差

    mean = np.tile((upper_bound + lower_bound) / 2.0, plan_horizon)
    var = np.tile(np.square(upper_bound - lower_bound) / 16, plan_horizon)

    while not done:
        # 用CEM优化动作序列
        actions_seq = cem.optimize(obs, mean, var)
        # 只执行第一步动作
        action = actions_seq[:action_dim]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_return += reward

        # 更新均值，用于下一次CEM迭代
        mean = np.concatenate([np.copy(actions_seq)[action_dim:], np.zeros(action_dim)])
        obs = next_obs

    return total_return

# ------------------ 运行测试 ------------------
return_list = []
for i in range(10):
    episode_return = run_mpc_episode(env, cem, plan_horizon, action_dim)
    print( 'episode {} return:{}'.format(i,episode_return) )
    return_list.append(episode_return)
print("MPC average return:", sum(return_list)/len(return_list))
