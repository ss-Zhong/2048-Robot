# Game without UI, only for Training

import tkinter as tk
import numpy as np
import random

class Game2048Env:
    def __init__(self, size=4, bot_mode=True):
        self.size = size
        self.bot_mode = bot_mode
        self.score = 0

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

        self.weights = np.array([
            [255, 127, 63, 63],
            [11, 15, 17, 19],
            [0, 0, 0, 0],
            [-3, -5, -7, -9]
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
                    board[i][j + 1] = 0
        return board, reward

    def reverse(self):
        return np.array([row[::-1] for row in self.board])

    def transpose(self):
        return np.transpose(self.board)

    def move_left(self):
        self.board = self.compress()
        self.board, reward = self.merge(self.board)
        self.board = self.compress()
        return reward

    def move_right(self):
        self.board = self.reverse()
        reward = self.move_left()
        self.board = self.reverse()
        return reward

    def move_up(self):
        self.board = self.transpose()
        reward = self.move_left()
        self.board = self.transpose()
        return reward

    def move_down(self):
        self.board = self.transpose()
        reward = self.move_right()
        self.board = self.transpose()
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
        if action == 0:
            reward = self.move_up()
        elif action == 1:
            reward = self.move_down()
        elif action == 2:
            reward = self.move_left()
        elif action == 3:
            reward = self.move_right()

        reward += np.sum((self.board - prev_board) * self.weights)

        if np.array_equal(prev_board, self.board):
            reward -= 10  # 惩罚无效动作
        else:
            self.add_new_tile()

        self.done = self.is_game_over()
        if self.done:
            reward -= 100

        reward = self.count_zero()

        return self.board, reward, self.done

    def get_board(self):
        """返回当前游戏状态的副本"""
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

    def count_zero(self):
        return np.sum(self.board == 0)
