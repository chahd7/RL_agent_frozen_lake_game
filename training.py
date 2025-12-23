import gymnasium as gym 
from agent import DQNAgent
from collections import deque
import numpy as np
import torch


#Initialization of the environment and the agent 
env = gym.make('FrozenLake-v1', reward_schedule=(1, 0, 0), is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n
agent = DQNAgent(state_size=state_size, action_size=action_size)

#training hyperparameters 
n_episodes = 2000 #max number of training episodes
max_t = 1000 #max number of timesteps per episode 
eps_start = 1.0 #starting value of epsilon 
eps_end = 0.01 #min value of epsilon 
eps_decay = 0.995 #factor for decreasing epsilon 

def train():
    #list to keep track the scores of each episode
    scores = []
    #last 100 scores 
    scores_window = deque(maxlen=100)
    #init epsilon 
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        state_temp, info = env.reset()



        state = np.zeros(state_size, dtype=np.float32)
        state[state_temp] = 1.0
        score = 0

        for t in range(max_t): 
            action = agent.act(state, eps)
            next_state_temp, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            #print(f"Moved to {next_state_temp} with reward: {reward}")

            next_state = np.zeros(state_size, dtype=np.float32)
            next_state[next_state_temp] = 1.0

            #make the agent do the action and learn from it 
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        #decrease the epsilon by picking the max 
        eps = max(eps_end, eps_decay * eps)

        #print the score for every 100 episodes
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')


        #check if env is solved 
        if np.mean(scores_window) >= 0.70:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            #save trained model weights 
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break

    return scores

scores = train()
env.close()

