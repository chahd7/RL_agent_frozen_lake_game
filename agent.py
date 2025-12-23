import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random 
from collections import deque, namedtuple

#neural network to aproximate the q-value function 
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialization of the network layers 

        :param state_size: number of features in the game (here 4: cart_position, cart_velocity, pole_angle and pole_angular_velocity)
        :param action_size: number of actions that the agent can take (here 2: 0 for left and 1 for right)
        """

        super(QNetwork, self).__init__() #initializes and allows this class to employ it so that the network functions well. Very important to always include or else the model will break
        #define the different layers that the model will have in a sequential way, one after the other
        self.network = nn.Sequential(
            nn.Linear(state_size, 128), #take the number of features and extract 128 hidden features out of it by learning combinations of state variables 
            nn.ReLU(), #for the introduction of nonlinearity which allows it to approximate complex functions and prevents it from collapsing into one single linear function
            nn.Linear(128, 128), #second hidden layer which refines the representation learned from the first one and helps capture even more complex relationships
            nn.ReLU(),
            nn.Linear(128, action_size) #takes as input the learned representation and outputs out of it a q value for each action 
        )

    def forward(self, state):
        """
        Definition of the forward pass of the network
        Takes state and returns the Q-value of the functions obtained by the assembled layers
        """
        return self.network(state)


#Hyperparameters
BUFFER_SIZE = 10000 #replay buffer size
BATCH_SIZE = 64 #minibatch size used for the training
GAMMA = 0.99 #discount factor for future rewards (here gives high importance to future rewards)
LR = 5e-4 #learning rate
UPDATE_EVERY = 4 #update the network after 4 steps 

#check if gpu available to train on else use the cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """The agent that will come to learn and interact with its environment and will use the QNetwork class as its network"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #make the qnetwork 
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) #decide how to change the neural network's weights with the goal of minimizing the loss

        #create the replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0 #used to keep track for the update


    #what happens after every action is taken 
    def step(self, state, action, reward, next_state, done):
        #append what was obtained to the buffer 
        self.memory.add(state, action, reward, next_state, done)

        #check if enough steps have passed to update the model 
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #check if there are enough experiences in the memory to learn from
            if len(self.memory) > BATCH_SIZE:
                #select a random sample of the memory to learn from 
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        #transform the state into a pytorch tensor for compability and add to it dimensionality to make it into a batch 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        #switch the model from training to evaluation mode 
        self.qnetwork_local.eval()

        #tell torch to not keep track of the gradient as we wont need it in the future 
        with torch.no_grad():
            #get the q values of the actions 
            action_values = self.qnetwork_local(state)

        #implemented the epsilon greedy approach. If less than epsilon get index of highest value from model else get at random 
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else: 
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma): 
        #unpack the experiences into the different component
        states, actions, rewards, next_states, dones = experiences

        #get the q value for the next state, make sure to detach to not backpropagate and then get the top value for the best action and expand the dimension
        Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)

        #implementation of the bellman equation. if episode ends then just reward else the full equation 
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

        actions = actions.long()

        #get expected q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #compute the loss between the targets and the expectations 
        loss = F.mse_loss(Q_expected, Q_targets)

        #minimize the loss 
        self.optimizer.zero_grad()  #clears old gradients
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    """Fixed sized buffer to store experience tuples"""
    def __init__(self, action_size, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """
        Sample a random batch of experience from memory to allow the model to learn from it
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    

    def __len__(self):
        """return current size of internal memory"""
        return len(self.memory)
    
        



