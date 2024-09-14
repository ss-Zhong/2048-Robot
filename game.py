# Game with UI

import tkinter as tk
import numpy as np
import random

class Game2048GUI:
    def __init__(self, size=4, bot_mode=False):
        self.size = size
        self.bot_mode = bot_mode
        self.score = 0

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

        self.root = tk.Tk()
        self.root.title("2048 Game")
        self.root.configure(bg="#000")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.outer_frame = tk.Frame(self.root, bg="#000")  # 设置为你想要的padding颜色
        self.outer_frame.grid(padx=10, pady=10)

        self.score_label = tk.Label(self.outer_frame, text=f"Score: {self.score}", font=("Times New Roman", 24, "bold"), bg="#000", fg="#fff")
        self.score_label.grid(row=0, column=0, padx=10, pady=10)

        # 内部框架，用来放置游戏内容
        self.grid_frame = tk.Frame(self.outer_frame, bg="#000")  # 实际的游戏框架
        self.grid_frame.grid(padx=10, pady=10)  # 内部的padding，可以根据需要调整 

        self.cells = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell_frame = tk.Frame(
                    self.grid_frame,
                    bg="#1f1f1f",
                    width=100,
                    height=100
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_label = tk.Label(cell_frame, text="", font=("Helvetica", 24, "bold"), fg="#776e65", width=4, height=2)
                cell_label.grid()
                row.append(cell_label)
            self.cells.append(row)

        self.update_gui()
        
        self.root.bind("<Key>", self.handle_keypress)

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def compress(self):       
        changeornot = False # 判断本操作是否造成移动
        new_board = np.zeros((self.size, self.size), dtype=int)

        for i in range(self.size):
            pos = 0
            for j in range(self.size):
                if self.board[i][j] != 0:
                    if pos != j:
                        changeornot = True

                    new_board[i][pos] = self.board[i][j]
                    pos += 1

        return new_board, changeornot

    def merge(self, board):
        changeornot = False # 判断本操作是否造成合并

        for i in range(self.size):
            for j in range(self.size - 1):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    board[i][j] *= 2
                    board[i][j + 1] = 0
                    self.score += board[i][j]
                    changeornot = True

        return board, changeornot

    def reverse(self):
        return np.array([row[::-1] for row in self.board])

    def transpose(self):
        return np.transpose(self.board)

    def move_left(self):
        self.board, changeornot1 = self.compress()
        self.board, changeornot2 = self.merge(self.board)
        self.board, changeornot3 = self.compress()
        return (changeornot1 or changeornot2 or changeornot3)

    def move_right(self):
        self.board = self.reverse()
        changeornot = self.move_left()
        self.board = self.reverse()
        return changeornot

    def move_up(self):
        self.board = self.transpose()
        changeornot = self.move_left()
        self.board = self.transpose()
        return changeornot

    def move_down(self):
        self.board = self.transpose()
        changeornot = self.move_right()
        self.board = self.transpose()
        return changeornot

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

    def update_gui(self):
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value == 0:
                    self.cells[i][j].config(text="", bg="#1f1f1f")
                else:
                    if value >= 8:
                        self.cells[i][j].config(text=str(value), fg="#f9f6f2", bg=self.get_color(value))
                    else:
                        self.cells[i][j].config(text=str(value), fg="#776e65", bg=self.get_color(value))

        self.score_label.config(text=f"Score: {self.score}")

    def get_color(self, value):
        colors = {
            2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179",
            16: "#F59563", 32: "#F67C5F", 64: "#F65E3B",
            128: "#EDCF72", 256: "#EDCC61", 512: "#EDC850",
            1024: "#EDC53F", 2048: "#EDC22E"
        }
        return colors.get(value, "#1f1f1f")

    def handle_keypress(self, event):
        key = event.keysym
        self.step(key)

    def step(self, direction):
        """机器人通过这个函数发送移动指令"""
        if direction == "Up" or direction == "w":
            changeornot = self.move_up()
        elif direction == "Down" or direction == "s":
            changeornot = self.move_down()
        elif direction == "Left" or direction == "a":
            changeornot = self.move_left()
        elif direction == "Right" or direction == "d":
            changeornot = self.move_right()

        if changeornot:
            self.add_new_tile()
            self.update_gui()
            if self.is_game_over():
                if self.bot_mode:
                    print("Game Over!")
                    self.root.quit()
                else:
                    self.show_game_over()
                    return 0 # 结束
            
            return 1 # 继续
        else:
            return -1 # 没变

    def get_board(self):
        """返回当前游戏状态的副本"""
        return np.copy(self.board)

    def start(self):
        """将事件循环放到单独的方法中"""
        self.root.mainloop()

    def show_game_over(self):
        # game_over_frame = tk.Frame(self.grid_frame, bg="#e2d6c9")
        # game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
        # game_over_label = tk.Label(game_over_frame, text="Game Over", font=("Times New Roman", 24), fg="#776e65", bg="#e2d6c9")
        # game_over_label.pack()
        self.score_label.config(text=f"Score: {self.score}", fg="#ff0000")

if __name__ == "__main__":
    game = Game2048GUI(4)
    game.start()  # 使用 start 方法启动游戏