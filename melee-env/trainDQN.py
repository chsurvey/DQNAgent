from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
from melee_env.agents.util import *

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQNAgent import DQNAgent

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default='C:/Users/USER/SSBM/ssbm.iso', type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

agent = DQNAgent()

players = [agent, CPU(enums.Character.KIRBY, 9)] #CPU(enums.Character.KIRBY, 5)]
env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)
device = agent.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
episodes = 300
episode_list = []
reward_list = []
env.start()

for episode in range(episodes):
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    total_reward = 0
    
    while not done:
        action = agent.select_action(gamestate)
        for i in range(len(players)):
            players[i].act(gamestate)    
        next_gamestate, done = env.step()
        obs, reward, done, info = agent.observation_space(next_gamestate)
        total_reward += reward
        
        reward = torch.tensor([reward], device=device)
        done = torch.tensor([1 if done else 0], device=device)
        state = agent.convert_state(gamestate)
        next_state = agent.convert_state(next_gamestate)

        agent.update(state, action, reward, next_state, done)
        gamestate = next_gamestate
    
    episode_list.append(episode + 1)
    reward_list.append(total_reward)
    plt.plot(episode_list, reward_list, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.pause(0.05)  # 잠시 대기 후 그래프 업데이트
