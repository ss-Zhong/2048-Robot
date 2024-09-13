import torch
import numpy as np
from game import Game2048GUI
from game_t import Game2048Env
from agent import DQNAgent
from tqdm import tqdm
import argparse

def train(env, agent, args):
    
    for episode in range(args.epoch):
        state = env.reset()
        state = state / 2048  # 归一化状态
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = next_state / 2048  # 归一化状态
            agent.store_transition(state, action, reward, next_state, done)

            agent.train()

            state = next_state
            total_reward += reward

            if done:
                print(f"[Episode {episode + 1}/{args.epoch}] Epsilon: {agent.epsilon:.8f}, \tTotal Reward: {total_reward}")

        if episode % args.update_target_frequency == 0:
            agent.update_target_model()

        agent.update_epsilon()

    torch.save(agent.model.state_dict(), "./model/bot_2048.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for Training 2048 Bot')
    parser.add_argument('--epoch', type=int, default=1000, help='num of epoch')
    parser.add_argument('--update_target_frequency', type=int,   default=10, help='update target frequency')
    args = parser.parse_args()

    env = Game2048Env()
    agent = DQNAgent(input_size=4, output_size=4) # 输入为 4x4 的棋盘，动作为 4 种
    train(env, agent, args)