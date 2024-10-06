# Game without UI, only for Training

import tkinter as tk
import numpy as np
import random

class Game2048Env:
    def __init__(self, size=4):
        self.size = size
        self.score = 0
        self.pre_score = 0

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

        self.weights = np.array([
            [8, 4, 2, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [-1, -2, -3, -4]
        ])

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
    
    def compress(self):
        new_board = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            pos = 0
            for j in range(self.size):
                if self.board[i][j] != 0:
                    new_board[i][pos] = self.board[i][j]
                    pos += 1
        return new_board

    def merge(self, board):
        reward = 0
        for i in range(self.size):
            for j in range(self.size - 1):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    board[i][j] *= 2
                    reward += board[i][j]
                    self.score += board[i][j]
                    board[i][j + 1] = 0
        return board, reward

    def move(self, action):
        self.board = np.rot90(self.board, action)

        self.board = self.compress()
        self.board, reward = self.merge(self.board)
        self.board = self.compress()

        self.board = np.rot90(self.board, -action)
        return reward

    def is_game_over(self):
        if 0 in self.board:
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
                if self.board[j][i] == self.board[j + 1][i]:
                    return False
        return True
        
    def step(self, action):
        prev_board = np.copy(self.board)

        # 0: left 1: up 2: right 3: down
        reward = self.move(action)

        if np.array_equal(prev_board, self.board) == False:
            self.add_new_tile()

        self.done = self.is_game_over()
        reward = self.get_reward()

        return self.board, reward, self.done

    def get_board(self):
        return np.copy(self.board)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.done = False
        self.add_new_tile()
        self.add_new_tile()
        return self.board

    def render(self):
        print(self.board)

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

    def get_reward(self):
        reward = np.sum(self.board == 0) / 2
        delta_score = self.score - self.pre_score
        if delta_score > 0:
            reward += np.log2(delta_score) / 2
            self.pre_score = self.score
        else:
            reward += 0
        
        # board_, _, _ = self.rearrange_board(self.board)
        # reward += np.sum(self.weights * self.preprocess_state(board_)) / 10
        # print(reward)

        return reward
    
        # 对state进行归一化操作
    def preprocess_state(self, state):
        state = np.where(state == 0, 1, state)
        return np.log2(state)
