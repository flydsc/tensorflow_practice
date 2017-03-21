import gym
import DQN
# import tensorflow as tf
# import numpy as np
# import random
# from collections import deque

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode


def main():
    env = gym.make(ENV_NAME)
    agent = DQN.DQN(env)
    for episode in range(EPISODE):
        state = env.reset()
        # train
        for step in range(STEP):
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            #Define reward
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                env.render()
                action = agent.action(state)
                state, reward, done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
            ave_reward = total_reward/TEST
            print 'episode', episode, 'Evaluation Average Reward:', ave_reward
            if ave_reward >= 200:
                break

if __name__ == '__main__':
    main()