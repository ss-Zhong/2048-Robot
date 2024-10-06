import torch
import torch.optim as optim
import numpy as np
from game import Game2048GUI
from game_t import Game2048Env
from agent import DQNAgent
from tqdm import tqdm
import argparse
import time
import random

def train(env, agent, args):
    
    for episode in tqdm(range(args.epoch_num)):
        state = env.reset()
        state = agent.one_hot_encode(state) # 对 state 进行归一化
        done = False
        total_reward = 0

        step_num = 0
        while not done:
            
            action = agent.act(env.board)
            # print(action, end='')
            next_state, reward, done = env.step(action)
            next_state = agent.one_hot_encode(next_state) # 对 next_state 进行归一化
            agent.store_transition(state, action, reward, next_state, done)

            agent.train()
            # env.render()
            # input()
            state = next_state
            total_reward += reward
            step_num += 1

            # if done:
            #     print(f"[Episode {episode + 1}/{args.epoch_num}] Epsilon: {agent.epsilon:.8f}, \tTotal Reward: {total_reward}")

        if episode % args.save_model_frequency == 0 and episode != 0:
            time_ = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
            torch.save(agent.model.state_dict(), f"./model/bot_2048_E{episode}_T{time_}.pth")
        
        if episode % 100 == 0:
            env.render()
            print(f'episode {episode} | score {env.score} | steps {step_num} | epsilon {agent.epsilon} | lr {agent.scheduler.get_last_lr()[0]}')

        agent.scheduler.step()

def set_seed(seed):
    random.seed(seed)  # 固定 Python 随机种子
    np.random.seed(seed)  # 固定 NumPy 随机种子
    torch.manual_seed(seed)  # 固定 PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # 固定 PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，固定所有 GPU 的种子
    torch.backends.cudnn.deterministic = True  # 保证每次卷积运算结果相同
    torch.backends.cudnn.benchmark = False  # 禁止动态选择最佳卷积算法

if __name__ == "__main__":
    # set_seed(0)

    parser = argparse.ArgumentParser(description='Parameters for Training 2048 Bot')
    parser.add_argument('--epoch_num', type=int, default=10000, help='num of epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--update_target_frequency', type=int, default=10, help='update target frequency')
    parser.add_argument('--save_model_frequency', type=int, default=1000, help='save model frequency')
    parser.add_argument('--model_path', type=str, default=None, help='load former model')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    env = Game2048Env()
    agent = DQNAgent(input_size = 4, output_size = 4, device = device,
                     batch_size=args.batch_size, learning_rate=args.learning_rate, 
                     update_target_frequency = args.update_target_frequency,
                     )
    if args.model_path is not None:
        agent.load_state_dict(torch.load(args.model_path))

    train(env, agent, args)