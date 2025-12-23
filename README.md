**RL Agent to learn how to play for the Frozen Lake Game**

The goal of this project is to build using Deep Q-Learning an agent that is able to consider the environment in which it operates and take optimal actions with the goal of maximizing a reward, making it a classic example of reinforcement learning in games.

The different elements are defined as follows: 
* Environment : The world where the agent operates
* Action : The move that the agent makes to get closer to its goal
* Reward : The Feedback from the environment after an action is taken. It can either be positive for a good action, or negative/none for bad actions.

The agent will therefore learn how to maximize the total reward over time by learning the policy, which is the strategy for choosing the best action in a given situation. 

To achieve that, Gymnasium, which is a popular python libray that provides game environments to teach and test RL algorithms, was used. It was chosen to work with the Frozen Lake game environment as it adds a little complexity to the more traditional use cases of Gymnasium. 
Additionally, PyTorch was used to help build the "brain" of the agent. 

<figure>
<img width="315" height="318" alt="image" src="https://github.com/user-attachments/assets/02168a9e-8e45-4d0c-9c3b-f2f3c6620945" />
<figcaption>Illustration of the Frozen Lake environment used for the agent to learn to solve</figcaption>
</figure>
