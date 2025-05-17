# 2048-Robot 🤖

![GitHub stars](https://img.shields.io/github/stars/ss-Zhong/2048-Robot?style=flat&color=5caaf3)
![Visits](https://badges.pufler.dev/visits/ss-Zhong/2048-Robot?color=47bdae&label=visits)
![License](https://img.shields.io/github/license/ss-Zhong/2048-Robot)
![Last commit](https://img.shields.io/github/last-commit/ss-Zhong/2048-Robot)

Play the game **2048** using a Deep Q-Network (DQN) agent.

## 🕹️Demo

<p align="center">
  <img src="img/README/demo.gif" alt="demo" width="40%">
</p>

## 🚀 Getting Started

- **🔧 Training the Agent on your own**

    To train the DQN agent from scratch:

    ```bash
    python train.py
    ```

- **🤖 Running the Bot**
    We provide the trained model at `/model/bot_final.pth`, which can be used to directly run and observe the bot's performance.
    ```
    python bot.py
    ```