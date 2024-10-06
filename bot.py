import time
import random
import threading
from game import Game2048GUI
from agent import DQNAgent
import numpy as np
import torch

class Bot2048:
    def __init__(self, game, agent = None):
        self.game = game
        self.agent = agent

    def choose_direction_random(self):
        """简单的机器人：随机选择方向"""
        directions = ["Up", "Down", "Left", "Right"]
        return random.choice(directions)

    def choose_direction_rl(self):
        bd = game.get_board()
        action = self.agent.act(bd)
        directions = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        # print(f"{directions[action]}  ", end='')
        return directions[action]

    def play(self):
        """机器人开始玩游戏"""
        over = 1
        while over != 0:
            if self.agent is None:
                direction = self.choose_direction_random()
            else:
                direction = self.choose_direction_rl()

            over = self.game.step(direction)
            # if over == 1:
            #     print(f"{direction}  ", end='')

            time.sleep(0.01)  # 让机器人慢一点，每半秒移动一次
        print("GAME OVER", end='')

def start_bot(game, AI = True):
    """启动机器人线程"""
    if AI:
        agent = DQNAgent(input_size = 4, output_size = 4, epsilon = 0)
        agent.model.load_state_dict(torch.load("./model/bot_2048_E2000_T20241006_1411.pth"))
        agent.model.eval()
    else:
        agent = None

    bot = Bot2048(game, agent = agent)
    bot.play()

if __name__ == "__main__":
    game = Game2048GUI()  # 创建游戏实例
    AI = True

    # 创建并启动机器人线程
    bot_thread = threading.Thread(target=start_bot, args=(game, AI))
    bot_thread.daemon = True  # 设置为守护线程，以便在主线程结束时自动退出
    bot_thread.start()

    # 启动游戏的主事件循环
    game.start()
