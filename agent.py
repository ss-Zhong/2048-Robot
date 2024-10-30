import torch
import random
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    '''
    Optimized DQN model class:
        - Contains convolution layers and fully connected layers.
        - Returns Q-values for each of the 4 actions (L, U, R, D).
    '''
    def __init__(self, in_channel):
        super(DQN, self).__init__()
        
        # First layer convolutional layers
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=(1,2))
        self.conv2 = nn.Conv2d(in_channel, 128, kernel_size=(2,1))

        # Second layer convolutional layers
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2,1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2,1))
        
        # Compute flattened shape based on input dimensions
        self.fc_input_dim = 4 * 3 * 128 * 2 + 2 * 4 * 128 * 2 + 3 * 3 * 128 * 2
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
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

        output = self.fc(concat)
        
        # Pass through fully connected layers to get Q-values
        return output

# class DQN(nn.Module):
#     '''
#     Optimized DQN model for 2048 game:
#         - Uses convolution layers for feature extraction.
#         - Fully connected layers for Q-value prediction for each action (L, U, R, D).
#     '''
#     def __init__(self, in_channel):
#         super(DQN, self).__init__()
        
#         # 第一层卷积层 (输入为 one-hot 编码的通道数)
#         self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
#         # 计算展平后的维度
#         self.fc_input_dim = 4 * 4 * 512  # 假设输入为 4x4 网格，卷积层输出通道为 512
        
#         # 全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 512),
#             nn.LeakyReLU(),  # 使用 LeakyReLU 代替 ReLU
#             nn.Linear(512, 4)  # 输出 4 个 Q 值，对应4个动作 (L, U, R, D)
#         )

#     def forward(self, x):
#         # 确保输入是 4D 张量，如果维度不够，则扩展维度
#         while x.dim() < 4:
#             x = x.unsqueeze(0)

#         # 卷积层前向传播
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))

#         # 展平卷积层的输出
#         x = x.view(x.size(0), -1)

#         # 全连接层前向传播，得到 Q 值
#         output = self.fc(x)

#         return output

class DQNAgent:
    def __init__(self, input_size, output_size, 
                 epsilon = 0.1, learning_rate = 0.0005, batch_size = 128, 
                 update_target_frequency = 10,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 model_path = None, one_hot_channel = 16):
        self.device = device

        self.model = DQN(one_hot_channel).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.target_model = DQN(one_hot_channel).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=50000) #################################50000
        self.gamma = 0.995  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_max = epsilon
        self.epsilon_min = 1e-4
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.step_total = 0
        self.one_hot_channel = one_hot_channel
        self.update_target_frequency = update_target_frequency
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=100, min_lr=1e-7)

        self.avg_loss = 0

    # 保存过去状态
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # agent 执行一步
    def act(self, board):

        # random
        if random.random() <= self.epsilon:
            possible_actions = self.possible_actions(board)
            return random.choice(possible_actions)  # 随机动作：上、下、左、右
        
        # model act
        else:
            # board, rotate, transpose = self.rearrange_board(board)

            state = self.one_hot_encode(board)
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state)

            possible_actions = self.possible_actions(board)
            actions = q_values.argsort()[0].cpu().numpy()[::-1]
            for action in actions:
                if action in possible_actions:
                    # if transpose:
                    #     dic = {0:1, 1:0, 2:3, 3:2}
                    #     action = dic[action]
                    # action = (action + rotate) % 4
                    return action
            
        return ValueError()

    def train(self, isDQN = True):
        if len(self.memory) < self.batch_size:
            return
        
        self.step_total += 1
        if self.step_total % self.update_target_frequency == 0:
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
        
        # rotate = random.randint(1, 3)
        # states_rotated = torch.rot90(states, k=rotate, dims=(2, 3))
        # actions_rotated = (actions - rotate) % 4
        # q_values_rotated = self.model(states_rotated).gather(1, actions_rotated)

        if isDQN:
            # DQN
            next_q_values = self.target_model(next_states).max(1)[0].detach()
        else:
            # DDQN
            next_action = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_action).squeeze().detach()

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values.squeeze(), target_q_values) #+ F.mse_loss(q_values_rotated.squeeze(), target_q_values)
        # loss /= 2
        # print("loss:", loss.item(), end='\t')
        self.avg_loss += loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()
        # print(loss.item())
        # input()

    def update_epsilon(self):
        # epsilon逐渐减小        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) / 30000
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def one_hot_encode(self, state, avgPos = False):
        '''
        one-hot encode (4, 4) numpy array to (16, 4, 4) numpy array
            - each channel 0..15 is a (4, 4) numpy array,
            - conatins 1's where original grid contains 2^i
            (first channel refers for empty tiles) 
        '''
        result = np.zeros((self.one_hot_channel, 4, 4), dtype=np.float32)
        if avgPos:
            empty_tiles = np.sum(state == 0)
            if empty_tiles > 0:
                possible = 1 / empty_tiles

            for i in range(4):
                for j in range(4):
                    if state[i][j] == 0:
                        result[0][i][j] = 1.0 - possible
                        result[1][i][j] = possible * 0.9 # 0.9概率出现2
                        result[2][i][j] = possible * 0.1 # 0.9概率出现4

                    else:
                        index = int(np.log2(state[i][j]))
                        result[index][i][j] = 1.0

        else:
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

    def rearrange_board(self, board):
        # print("Former\n", board)

        blocks = {
            0: board[0:2, 0:2],
            1: board[0:2, 2:4],
            2: board[2:4, 2:4],
            3: board[2:4, 0:2]
        }
        sums = {key: np.sum(block) for key, block in blocks.items()}

        # Step 2: 按照块的总和对块进行排序
        sorted_blocks = sorted(sums.items(), key=lambda x: x[1], reverse=True)
        rotate = sorted_blocks[0][0] # 第一
        second_block_key = sorted_blocks[1][0]
        third_block_key = sorted_blocks[2][0]

        if rotate != 0:
            board = np.rot90(board, rotate)
            second_block_key = (second_block_key - rotate) % 4
            third_block_key = (third_block_key - rotate) % 4
        
        transpose = (second_block_key == 2 and third_block_key == 3) or second_block_key == 3
        if transpose:
            board = np.transpose(board)
        # print("After\n", board)
        return board, rotate, transpose