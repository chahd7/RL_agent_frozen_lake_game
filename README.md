**RL Agent to learn how to play for the Frozen Lake Game**

The goal of this project is to build using Deep Q-Learning an agent that is able to consider the environment in which it operates and take optimal actions with the goal of maximizing a reward, making it a classic example of reinforcement learning in games.

The different elements are defined as follows: 
* Environment : The world where the agent operates
* Action : The move that the agent makes to get closer to its goal
* Reward : The Feedback from the environment after an action is taken. It can either be positive for a good action, or negative/none for bad actions. Here the reward is such that it gives +1 when the goal is reached and 0 for when it moves to an ice block or falls into a hole. 

The agent will therefore learn how to maximize the total reward over time by learning the policy, which is the strategy for choosing the best action in a given situation. 

To achieve that, Gymnasium, which is a popular python libray that provides game environments to teach and test RL algorithms, was used. It was chosen to work with the Frozen Lake game environment as it adds a little complexity to the more traditional use cases of Gymnasium. 
Additionally, PyTorch was used to help build the "brain" of the agent. 

<figure>
  <p align="center">
    <img width="315" height="318" alt="image" src="https://github.com/user-attachments/assets/02168a9e-8e45-4d0c-9c3b-f2f3c6620945" />
    <figcaption>Illustration of the Frozen Lake environment used for the agent to learn to solve</figcaption>
  </p>
</figure>

At each start of the game, the player is located at position (0,0) while the goal is located at position (3,3) in the case of a 4x4 grid. The holes to be avoided are positioned in a specific manner that remains constant. It was chosen to work with is_slippery=True which allows to model a more natural movement from the player and allows it to move perpendicular to the intended direction sometimes. 

The action space is defined such that it can take on 4 values:
* 0 : Move left
* 1 : Move down
* 2 : Move right
* 3 : Move up

The observation space on the other hand, can take any integer value between 0 and 16, and it calculated using current_row * ncols + current_col (where both the row and col start at 0). For a 4x4 map, the goal is therefore located at 3 * 4 + 3 = 15. Since it returns an integer that can't be processed directly by the network, a zero array of the size of the observation space is created with the index at which the player is located set to 1.0.

A QNetwork class was first created, which instantiates the nn.Module class present in PyTorch, and which represents the base class for all neural network modules. This class initializes the network and creates the different layers that constitute it. The architecture is as follows: 

<figure>
  <p align="center">
    <img width="1053" height="234" alt="image" src="https://github.com/user-attachments/assets/e5562221-300d-4e3d-8ecd-4f4c1b437ede" />
    <figcaption>Illustration of architecture of the Neural Network</figcaption>
  </p>
</figure>

The model therefore takes the different positions where the user can be located, and through a combination of Linear layers, which apply linear transformations to the data, and the ReLU activation function which keeps positive results while clipping negative ones, the network take as input the location of the player and outputs a q-value for each different action that can be taken. 

A second class labeled DQNAgent will use this QNetwork class and set the optimizer to be an Adam optimizer, which will be used to adjust learning rates during the training. Different methods for the agent were then defined such as: 
* step() which will come to append the results obtained after the agent has done an action to the ReplayBuffer and then check if enough steps have passed to update the model. If it is the case, it checks if the memory has enough samples and then picks a random one to learn from it.
* act() which implements an epsilon greedy approach. A random number is picked and then it is bigger than epsilon, the action with the higher Q-value is picked. Else a random action is picked. As the model learns further, epsilon is decreated meaning that it takes less random actions and takes more actions that have the highest Q-value. 
* learn() which will unpack the different elements that constitute one of the experiences saved int he ReplayBuffer and then using the network get the max predicted Q-Value for the next state. Using this value, Q-target is computed such that Q_target = rewards + (gamma + Q_target_next * (1-dones)) and will therefore be only equal to the rewards if dones=True. It then gets the actual Q-value from the current state from the model and computes the MSE loss by having Q_target - Q_actual. The gradients of the loss with respect to the model parameters are then computed and the optimizer moves the parameters in the direction which will minimize the loss.

The ReplayBuffer class on the other end is created to keep track of the different experiences and provides the different methods to allow the DQNAgent class to be able to leverage it. An experience is defined as a tuple that contains the elements: state, action, reward, next_state and done. A new experience can then be created and added to the buffer and a sample of experiences can be taken to allow the agent to learn on it. 

A training loop was then created that uses the DQNAgent class to create the agent. For each episode and for each timestep per episode, the agent gives an action that the environment will perform before returning the next_state, reward, truncated and terminated elements, which are then saved as sample to the ReplayBuffer. The state is then set to the next_state. With done being set to truncated or terminated, as soon as one of them becomes true the episodde ends. The score is then appended. As threshold, 0.70 was chosen as it showcases good performance with is_slippery=True and once the average score is reached, the model weights are saved. 








