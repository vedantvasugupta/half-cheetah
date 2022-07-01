from pyrsistent import b
from classes import Agent
import gym
from gym import wrappers
import numpy as np
import os
import pickle


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    env = gym.make('HalfCheetah-v4')
    agent = Agent(alpha = 0.0001, beta =0.001, input_dims = [17], tau=0.001, env= env, n_actions = 6)
    np.random.seed(0)
    
    # agent.load_models()
    score_history = []
    
    max_episode_steps = 2000
    best_score = float('-inf')
    for i in range(5000):
        obs = env.reset()
        done = False
        score = 0
        for t in range(max_episode_steps):
            act = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_obs, done)
            agent.learn()
            score += reward
            obs = new_obs
            # env.render()
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        if not done: 
            print('Timed out')
        score_history.append(score)
        if score > best_score:
            best_score = score
        print('episode: ', i, 'score: ', score, 'trailing 100 games avg: ', np.mean(score_history[-100:]), ' best score: ',best_score)
        if i!= 0:
            if i % 25 == 0:
                agent.save_models()
                with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([score_history,], f)
