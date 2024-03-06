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
from NstepDQNAgent import NstepDQNAgent

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="/home/vlab/SSBM/ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

agent = NstepDQNAgent(n_step=60) #DQNAgent()
#agent.policy_net = torch.load("/home/vlab/SSBM/melee-env/model_180.pth", map_location=agent.device)
#agent.target_net = torch.load("/home/vlab/SSBM/melee-env/model_180.pth", map_location=agent.device)

players = [agent, CPU(enums.Character.FOX, 9)] #CPU(enums.Character.KIRBY, 5)]
env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)
device = agent.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
episodes = 1000
episode_list = []
reward_list = []
env.start()

t = 0

for episode in range(episodes):
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    total_reward = 0
    
    while not done:
        action = agent.select_action(gamestate)
        # print(action)
        players[0].apply(action)
        players[1].act(gamestate)
          
        next_gamestate, done = env.step()
        obs, reward, done, info = agent.observation_space(next_gamestate)
        
        if done == True:
            reward = -5 # die
            
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        d = torch.tensor([1 if done else 0], device=device)
        state = agent.convert_state(gamestate)
        next_state = agent.convert_state(next_gamestate)
        agent.update(state, action, reward, next_state, d)
        
        if abs(reward) == 5:
            agent.nstep_buffer.clear()
            episode_list.append(t + 1)
            reward_list.append(total_reward)
            plt.plot(episode_list, reward_list, color='blue')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.pause(0.05)
            t += 1
            total_reward = 0
        
        gamestate = next_gamestate
    
    if episode % 30 == 0:
        torch.save(agent.policy_net, f'/home/vlab/SSBM/nstepDQN/{agent.n_step}step_{episode}.pth')
        print(reward_list[-1])
        plt.savefig(f'/home/vlab/SSBM/nstepDQN/{agent.n_step}step.png')
