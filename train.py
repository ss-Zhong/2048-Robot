import torch
import torch.optim as optim
import numpy as np
from game_t import Game2048Env
from agent import DQNAgent
from tqdm import tqdm
import argparse
import time
import random
import wandb

def train(env, agent, args):
    
    for episode in tqdm(range(args.epoch_num)):
        state = env.reset()
        state = agent.one_hot_encode(state, avgPos = False) # 对 state 进行归一化
        done = False
        
        step_num = 0
        total_reward = 0
        while not done:
            
            action = agent.act(env.board)
            next_state = env.step(action)
            # next_state = agent.one_hot_encode(next_state, avgPos = False) # 对 next_state 进行归一化
            next_state_with_newtile, reward, done = env.step_tile()
            max_ = np.max(next_state_with_newtile)
            next_state_with_newtile = agent.one_hot_encode(next_state_with_newtile, avgPos = False) 

            # print(ratio)

            if episode <= 50 or random.random() < 0.5 or max_ >= 1024:
                agent.store_transition(state, action, reward, next_state_with_newtile, done)
            
            agent.train()
            state = next_state_with_newtile
            step_num += 1

            total_reward += reward

        wandb.log({"reward": total_reward, "score": env.score})

        if (episode+1) % args.save_model_frequency == 0:
            time_ = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
            torch.save(agent.model.state_dict(), f"./model/bot_2048_E{episode+1}_T{time_}.pth")
        
        if (episode+1) % 10 == 0:
            if agent.step_total != 0:
                agent.avg_loss = agent.avg_loss / agent.step_total
            else:
                agent.avg_loss = 0

            if (episode+1) % 100 == 0:
                env.render()
                print(f'episode {episode+1} | score {env.score} | steps {step_num} | epsilon {agent.epsilon} | lr {agent.scheduler.get_last_lr()[0]} | loss {agent.avg_loss}')
            wandb.log({"lr": agent.scheduler.get_last_lr()[0], "loss": agent.avg_loss, "step": agent.step_total})
            agent.avg_loss = 0
            agent.step_total = 0

        agent.scheduler.step(agent.avg_loss)

def test(env, agent, args):

    best_score = 0
    
    for episode in tqdm(range(args.epoch_num)):
        state = env.reset()
        state = agent.one_hot_encode(state, avgPos = False) # 对 state 进行归一化
        done = False

        total_reward = 0
        while not done:
            
            action = agent.act(env.board)
            env.step(action)
            next_state_with_newtile, reward, done = env.step_tile()
            next_state_with_newtile = agent.one_hot_encode(next_state_with_newtile, avgPos = False) 
            state = next_state_with_newtile

        if env.score > best_score:
            best_score = env.score
            print(f"Epoch-{episode} SCORE: {best_score}\n{env.board}")
        
        wandb.log({"score": env.score})

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
    parser.add_argument('--epoch_num', type=int, default=1000, help='num of epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--update_target_frequency', type=int, default=10, help='update target frequency')
    parser.add_argument('--save_model_frequency', type=int, default=1000, help='save model frequency')
    parser.add_argument('--model_path', type=str, default=None, help='load former model')
    parser.add_argument('--mode', type=str, default='train', help='load former model')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon')
    parser.add_argument('--one_hot_channel', type=int, default=16, help='Channel of One-hot Encode')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    wandb.init(
        # set the wandb project where this run will be logged
        project="2048-robot",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "architecture": "DQN",
        "epochs": args.epoch_num,
        }
    )

    env = Game2048Env()
    agent = DQNAgent(input_size = 4, output_size = 4, device = device,
                     batch_size = args.batch_size, learning_rate = args.learning_rate, 
                     update_target_frequency = args.update_target_frequency,
                     model_path = args.model_path,
                     epsilon = args.epsilon,
                     one_hot_channel = args.one_hot_channel
                     )
    
    if args.mode == 'train':
        train(env, agent, args)
    else:
        test(env, agent, args)

    wandb.finish()