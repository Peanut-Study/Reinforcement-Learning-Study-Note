'''
环境：倒立摆，具体介绍见https://gymnasium.farama.org/environments/classic_control/pendulum/
一个倒立摆，目标是让他竖直向上，偏离数值向上的角度记为theta属于[-pi,pi],往左偏为正，往右偏为负数，即逆时针为正。
杆的长度是1米。
状态，也就是观测，包括3个数，杆自由端的坐标x,y和角速度.坐标采用的是平面直角坐标系
x=cos(theta),y=sin(theta),角速度Angular Velocity属于[-8,8]
动作，就是对这个杆施加一个扭矩，torque，属于[-2.2],逆时针为正。
奖励函数=-(theta的平方+0.1*角速度的平方+0.001*扭矩的平方)
到达200步时直接停止。
这个游戏中途如果没有成功，则到达200步时停止，不会中途停止。
'''
import gymnasium as gym
# 创建环境
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode='human')  # render_mode='human' 可以可视化

# 可以来看看他的状态空间和动作空间
print("观察空间:", env.observation_space)
# Box([-1. -1. -8.], [1. 1. 8.], (3,), float32) Box表示连续的,[下界，上界，维度是3，float32]
print("动作空间:", env.action_space)
# Box(-2.0, 2.0, (1,), float32) (下届，上界，维度是1，float32)

# 重置环境
obs, info = env.reset(seed=0)
print("初始观测:", obs)
# 观测由3个数构成，自由端x坐标，y坐标（笛卡尔坐标，单位为米），角速度
# 比如 [ 0.6520163   0.758205   -0.46042657] ,第一个就是x,第二个是y,第三个是角速度

# 随机采用一个动作
action = env.action_space.sample()  # 随机动作
print(action)  #[1.7855948],表示施加的扭矩

# 可视化演示
obs, info = env.reset(seed=0)
for i in range(250):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
