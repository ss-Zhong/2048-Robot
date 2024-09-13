import torch
import random
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size = 4, output_size =  4):
        super(DQN, self).__init__()

        self.input_size = input_size
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        
        # 定义全连接层
        self.fc1 = nn.Linear(128, 128)  # 4x4 是棋盘的尺寸
        self.fc2 = nn.Linear(128, output_size)  # 输出动作的 Q 值

    def forward(self, x):
        x = x.view(-1, 1, self.input_size, self.input_size)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, epsilon = 1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice([0, 1, 2, 3])  # 随机动作：上、下、左、右
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 从memory中采样一个batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表转换为单个 numpy.array 然后再转为 PyTorch Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Q值计算与更新
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # epsilon逐渐减小
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
