import gymnasium as gym
import time 
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from agent import DQNAgent
import numpy as np
import torch


env = gym.make('FrozenLake-v1', render_mode="human", reward_schedule=(1, 0, 0), is_slippery = True)
state_size = env.observation_space.n
action_size = env.action_space.n 

agent = DQNAgent(state_size=state_size, action_size=action_size)

agent.qnetwork_local.load_state_dict(torch.load("checkpoint.pth"))

episode_to_watch = 10
total_completed = 0


for episode in range(episode_to_watch):
    env = gym.make('FrozenLake-v1', render_mode="human", reward_schedule=(1, 0, 0), is_slippery = True)

    state_size = env.observation_space.n
    action_size = env.action_space.n 
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.qnetwork_local.load_state_dict(torch.load("checkpoint.pth"))


    state_temp, info = env.reset()

    state = np.zeros(state_size, dtype=np.float32)
    state[state_temp] = 1.0

    done = False 
    reward = 0
    episode_reward = 0

    print(f"Watching for Episode {episode + 1}")


    while not done:
        #render the environment 
        env.render()

        #make the agent pick an action based on the state
        action = agent.act(state=state)

        #perform the action on the environment 
        observation, reward, terminated, truncated, info = env.step(action)

        #transform the next state into array 
        next_state = np.zeros(state_size, dtype=np.float32)
        next_state[observation] = 1.0

        done = terminated or truncated
        episode_reward += reward

        state = next_state

        time.sleep(0.02)


    if episode_reward == 1:
        print("Episode completed successfully")
        total_completed += 1





print(f"Completed successfully {total_completed} episodes! ")
env.close()