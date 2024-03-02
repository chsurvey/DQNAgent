from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import argparse
from DQNAgent import DQNAgent
from melee_env.agents.util import *

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import argparse

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="/home/vlab/SSBM/ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

agent = DQNAgent()
players = [agent, CPU(enums.Character.KIRBY, 9)] #CPU(enums.Character.KIRBY, 5)]

env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)

episodes = 2; reward = 0
env.start()

observation_space = ObservationSpace()
action_space = ActionSpace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for episode in range(episodes):
    # gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    state, done = env.setup(enums.Stage.BATTLEFIELD)
    while not done:
        for i in range(len(players)):
            players[i].act(state)
        obs, reward, done, info = agent.observation_space(state)
        state, done = env.step()
