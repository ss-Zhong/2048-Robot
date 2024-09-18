import torch
import random
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# class DQN(nn.Module):
#     def __init__(self, input_size = 4, output_size = 4): 
#         super(DQN, self).__init__()
#         self.input_size = input_size
#         self.conv1 = nn.Conv2d(16, 64, kernel_size=2, stride=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
#         self.fc1 = nn.Linear(128 * 2 * 2, 128)
#         self.fc2 = nn.Linear(128, output_size)

#     def forward(self, x):
#         # Ensure input is 4D, if less than 4D, unsqueeze to add dimensions
#         while x.dim() < 4:
#             x = x.unsqueeze(0)  # Add a dimension at the front

#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)  # 展平
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class DQN(nn.Module):
    '''
    Optimized DQN model class:
        - Contains convolution layers and fully connected layers.
        - Returns Q-values for each of the 4 actions (L, U, R, D).
    '''
    def __init__(self):
        super(DQN, self).__init__()
        
        # First layer convolutional layers
        self.conv1 = nn.Conv2d(16, 128, kernel_size=(1,2))
        self.conv2 = nn.Conv2d(16, 128, kernel_size=(2,1))

        # Second layer convolutional layers
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2,1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2,1))
        
        # Compute flattened shape based on input dimensions
        self.fc_input_dim = 4 * 3 * 128 * 2 + 2 * 4 * 128 * 2 + 3 * 3 * 128 * 2
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # Ensure input is 4D, if less than 4D, unsqueeze to add dimensions
        while x.dim() < 4:
            x = x.unsqueeze(0)  # Add a dimension at the front

        # First layer of convolutions
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        
        # Second layer of convolutions
        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))
        
        # Flatten and concatenate layers
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x2 = x2.view(x2.size(0), -1)  # Flatten
        x11 = x11.view(x11.size(0), -1)
        x12 = x12.view(x12.size(0), -1)
        x21 = x21.view(x21.size(0), -1)
        x22 = x22.view(x22.size(0), -1)

        # Concatenate all flattened outputs
        concat = torch.cat((x1, x2, x11, x12, x21, x22), dim=1)
        
        # Pass through fully connected layers to get Q-values
        return self.fc(concat)

class DQNAgent:
    def __init__(self, input_size, output_size, 
                 epsilon = 0.1, learning_rate = 0.0005, batch_size = 128, 
                 update_target_frequency = 10,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=10000) #################################50000
        self.gamma = 0.99  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_max = epsilon
        self.epsilon_min = 0.0001
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.step_total = 0
        self.update_target_frequency = update_target_frequency
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 保存过去状态
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # agent 执行一步
    def act(self, board):

        state = self.one_hot_encode(board)
        possible_actions = self.possible_actions(board)

        if random.random() <= self.epsilon:
            return random.choice(possible_actions)  # 随机动作：上、下、左、右
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)

        # print(q_values)
        actions = q_values.argsort()[0].cpu().numpy()[::-1]
        for action in actions:
            if action in possible_actions:
                return action
        
        return ValueError()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.step_total += 1
        if self.step_total % self.update_target_frequency == 0:
            self.step_total = 0
            self.update_target_model()

        # 从memory中采样一个batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表转换为单个 numpy.array 然后再转为 PyTorch Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q值计算与更新
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()
        # print(loss.item())
        # input()

    def update_epsilon(self):
        # epsilon逐渐减小        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) / 10000
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def one_hot_encode(self, state):
        '''
        one-hot encode (4, 4) numpy array to (16, 4, 4) numpy array
            - each channel 0..15 is a (4, 4) numpy array,
            - conatins 1's where original grid contains 2^i
            (first channel refers for empty tiles) 
        '''
        result = np.zeros((16, 4, 4), dtype=np.float32)

        for i in range(4):
            for j in range(4):
                if state[i][j] == 0:
                    result[0][i][j] = 1.0
                else:
                    index = int(np.log2(state[i][j]))
                    result[index][i][j] = 1.0

        return result
    
    def _can_perform(self, action, board):
        tmp = np.rot90(board, action)
        for i in range(4):
            empty = False
            for j in range(4):
                empty |= tmp[i, j] == 0
                if tmp[i, j] != 0 and empty:
                    return True
                if j > 0 and tmp[i, j] != 0 and tmp[i, j] == tmp[i, j-1]:
                    return True
        return False
    
    # returns a list of all possible actions
    def possible_actions(self, board):
        res = []
        for action in range(4):
            if self._can_perform(action, board):
                res.append(action)
        return res