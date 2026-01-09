'''
CartPole-v1 环境介绍：
https://gymnasium.farama.org/environments/classic_control/cart_pole/
状态有4个参数，车的位置，车的速度，杆偏离中间的角度，杆的角速度
车的位置，[-4.8,4.8],连续变量
车的速度，负无穷到正无穷
杆偏离中间的角度，[-24°,24°]
杆的角速度，负无穷到正无穷
动作有两个，左和右。
奖励，每坚持1步，获得1奖励
起始状态，所有观测值在 （-0.05， 0.05） 中均被赋予一个均匀随机值。
游戏结束：
1，当车的位置超过[-2.4,2.4],游戏结束
2，当杆的角度超过[-12°，12°],游戏结束
3，当坚持500步，游戏成功，游戏结束
'''
import gymnasium as gym

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")  # 显示画面
state, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()   # 随机动作，仅用于展示
    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Game Over!")
        break

env.close()