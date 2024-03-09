from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
from melee_env.agents.util import *

import time
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

from tqdm import tqdm
import psutil
import sys

from DQNAgent import DQNAgent

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="./ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

agent = DQNAgent()

players = [agent, CPU(enums.Character.FOX, 1)] #CPU(enums.Character.KIRBY, 5)]
env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)
device = agent.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
episodes = 100
episode_list = []
reward_list = []

freq = 3
save_name = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())

env.start()
for lvl in range(1,10):
    env.players[1].lvl = lvl
    for episode in tqdm(range(episodes)):
        if episode % freq == freq-1:
            for proc in psutil.process_iter():
                if proc.name() == "Slippi Dolphin.exe":
                    parent_pid = proc.pid
                    parent = psutil.Process(parent_pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
            players = [agent, CPU(enums.Character.FOX, lvl)] #CPU(enums.Character.KIRBY, 5)]
            env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)
            # agent.policy_net.load_state_dict(torch.load(save_name + ".pth"))
            # agent.target_net.load_state_dict(agent.policy_net.state_dict())
            env.start()
        
        print("==>setting env...")
        gamestate, done = env.setup(enums.Stage.FINAL_DESTINATION)
        print("done.")
        total_reward = 0
        
        while not done:
            action = agent.select_action(gamestate)
            # print(action)
            players[0].apply(action)
            players[1].act(gamestate)       
            
            next_gamestate, done = env.step()
            obs, reward, done, info = agent.observation_space(next_gamestate)
            total_reward += reward
            
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([1 if done else 0], device=device)
            state = agent.convert_state(gamestate)
            next_state = agent.convert_state(next_gamestate)

            agent.update(state, action, reward, next_state, done)
            gamestate = next_gamestate
            #print(len(agent.replay_buffer))
        
        episode_list.append(episode + 1)
        reward_list.append(total_reward)
        
        if episode % 30 == 29:
            torch.save(agent.policy_net, f'./model/DQN_{episode+1}.pth')
            print(reward_list[-1])
            plt.figure()
            plt.plot(episode_list, reward_list, color='blue')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.savefig(f'reward_plot.png')
            plt.close()
    print("episode",episode+1,"ended.")
