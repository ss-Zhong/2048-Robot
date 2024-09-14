import torch
import numpy as np
from game import Game2048GUI
from game_t import Game2048Env
from agent import DQNAgent
from tqdm import tqdm
import argparse
import time



def train(env, agent, args):
    
    for episode in tqdm(range(args.epoch_num)):
        state = env.reset()
        state = agent.one_hot_encode(state) # 对 state 进行归一化
        done = False
        total_reward = 0

        step_num = 0
        while not done:
            
            action = agent.act(state)
            # print(action, end='')
            next_state, reward, done = env.step(action)
            next_state = agent.one_hot_encode(next_state) # 对 next_state 进行归一化
            agent.store_transition(state, action, reward, next_state, done)

            agent.train()

            state = next_state
            total_reward += reward
            step_num += 1

            # if done:
            #     print(f"[Episode {episode + 1}/{args.epoch_num}] Epsilon: {agent.epsilon:.8f}, \tTotal Reward: {total_reward}")

        agent.update_epsilon()

        if episode % args.update_target_frequency == 0:
            agent.update_target_model()

        if episode % args.save_model_frequency == 0 and args.save_model_frequency != 0:
            torch.save(agent.model.state_dict(), f"./model/bot_2048_E{episode}_T{time.time()}.pth")
        
        if episode % 100 == 0:
            env.render()
            print(f'episode {episode} | score {env.score} | steps {step_num}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for Training 2048 Bot')
    parser.add_argument('--epoch_num', type=int, default=10000, help='num of epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--update_target_frequency', type=int, default=10, help='update target frequency')
    parser.add_argument('--save_model_frequency', type=int, default=1000, help='save model frequency')
    parser.add_argument('--model_path', type=str, default=None, help='load former model')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    env = Game2048Env()
    agent = DQNAgent(input_size = 4, output_size = 4, device = device) # 输入为 4x4 的棋盘，动作为 4 种
    if args.model_path is not None:
        agent.load_state_dict(torch.load(args.model_path))

    train(env, agent, args)