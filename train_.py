import torch
import numpy as np
from game import Game2048GUI
from game_t import Game2048Env
from agent import DQNAgent
from tqdm import tqdm
import argparse
import time
import random

import torch.nn.functional as F
import torch.nn as nn
from itertools import count

class ReplayMemory(object):
    '''
    Experience replay memory for DQN training.
        - a circular array of (using deque) Transitions:
        - each trainsition is a single experience
        - sample random batches from the experiences deque
    '''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    '''
    DQN model class.
        - contains convolution layers and
          fully-connected layers.
        - returns Q-values for each one
          of the 4 actions (L, U, R, D)
        - both policy and target network
          are instances of this class.
    '''
    def __init__(self):
        super(DQN, self).__init__()
        
        # first layer conv. layers, recieve one-hot
        # encoded (16, 4, 4) array as input
        self.conv1 = nn.Conv2d(16, 128, kernel_size=(1,2))
        self.conv2 = nn.Conv2d(16, 128, kernel_size=(2,1))

        # second layer conv. layers, recieve first layer as input
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2,1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2,1))
        
        # flattened shape
        first_layer = 4*3*128*2
        second_layer = 2*4*128*2 + 3*3*128*2
        self.fc = nn.Sequential(
            nn.Linear(first_layer + second_layer, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.to(device)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        
        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))
        
        # flatten and concat layers, input for linear layer
        s1 = x1.shape
        s2 = x1.shape
        
        s11 = x11.shape
        s12 = x12.shape
        s21 = x21.shape
        s22 = x22.shape
        
        x1 = x1.view(s1[0], s1[1]*s1[2]*s1[3])
        x2 = x2.view(s2[0], s2[1]*s2[2]*s2[3])

        x11 = x11.view(s11[0], s11[1]*s11[2]*s11[3])
        x12 = x12.view(s12[0], s12[1]*s12[2]*s12[3])
        x21 = x21.view(s21[0], s21[1]*s21[2]*s21[3])
        x22 = x22.view(s22[0], s22[1]*s22[2]*s22[3])
        
        concat = torch.cat((x1,x2,x11,x12,x21,x22), dim=1)
        return self.fc(concat)
    
# hyperparameters
BATCH_SIZE = 128      # batch size sampled from memory
GAMMA = 0.99          # discount rate
EPS_START = 0.1       # initial epsilon
EPS_END = 0.0001      # final epsilon
BUFFER_SIZE = 50000   # replay memory size
UPDATE_RATE = 10      # rate of policy updates to target
LEARNING_RATE = 1e-4  # learning rate alpha
DECAY = 10000         # epsilon decay

# use gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(BUFFER_SIZE)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
steps_done = 0
epsilon = EPS_START

# save training results
episode_durations = []
scores = []
max_tiles = []

def one_hot_encode(state):
    '''
    one-hot encode (4, 4) numpy array to (16, 4, 4) numpy array
        - each channel 0..15 is a (4, 4) numpy array,
        - conatins 1's where original grid contains 2^i
          (first channel refers for empty tiles) 
    '''
    result = np.zeros((1, 16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            result[0][state[i][j]][i][j] = 1.0

    # print("re:",result)
    # input()
    return result

def select_action(state, possible_actions, train=True):
    # epsilon-greedy
    global epsilon
    state = torch.from_numpy(state).to(device)
    # explore
    if random.random() < epsilon and train:
        return np.random.choice(possible_actions)
    # exploit
    actions = policy_net(state).argsort()[0].cpu().numpy()[::-1]
    for action in actions:
        if action in possible_actions:
            return action
    raise ValueError()

def optimize_model():
    global steps_done, epsilon
    if len(memory) < BATCH_SIZE:
        return
    steps_done += 1
    # update target to policy if needed
    if steps_done % UPDATE_RATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # sample batch from memory
    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*memory.sample(BATCH_SIZE))
	
    batch_state = torch.FloatTensor(np.array(batch_state)).squeeze(1).to(device)
    batch_next_state = torch.FloatTensor(np.array(batch_next_state)).squeeze(1).to(device)
    batch_action = torch.Tensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.Tensor(batch_reward).unsqueeze(1).to(device)
    batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
 
    # compute loss & optimize model
    # next states are relevant only if game isn't done
    with torch.no_grad():
        next_q_vals = target_net(batch_next_state)
        y = batch_reward + (1 - batch_done) * GAMMA * torch.max(next_q_vals, dim=1, keepdim=True)[0]
        x = policy_net(batch_state).gather(1, batch_action.long())

        # print(y, x)
    
    loss = F.mse_loss(policy_net(batch_state).gather(1, batch_action.long()), y)
    print(loss.item())
    input()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decrease epsilon
    if epsilon > EPS_END:
        epsilon -= (EPS_START - EPS_END) / DECAY

def train(env: Env, episodes=3000, save_rate=100, print_rate=100):
    global steps_done, epsilon
    start_time = time.time()
    for episode in range(episodes):
        state = one_hot_encode(env.reset())
        for t in count():
            # select and make a single action
            action = select_action(state, env.possible_actions())
            reward = env.step(action)
            next_state = one_hot_encode(env.state())
            done = env.is_done()

            # add experience to memory and optimize
            memory.push(state, next_state, torch.FloatTensor([[action]]).to(device), torch.FloatTensor([[reward]]).to(device), torch.FloatTensor([[done]]).to(device))
            optimize_model()

            state = next_state
            if done:
                episode_durations.append(t + 1)
                scores.append(env.score())
                max_tiles.append(env.max_tile())
                break
        
        # save model weights
        if episode % save_rate == 0:
            torch.save(policy_net.state_dict(), './model/policy_net.pth')
            torch.save(target_net.state_dict(), './model/target_net.pth')
        
        # show current progress
        if episode % print_rate == 0:
            env.render()
            print(f'episode {episode} | score {scores[-1]} | max {max_tiles[-1]} | steps {episode_durations[-1]} | time {time.time()-start_time}')
    print("Training Completed.")

env = Game2048Env()
train(env)