import torch
import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import time
import torch.nn as nn # Need this to define the architecture
import ale_py   # ← IMPORTANT LINE to ensure Atari environments work


gym.register_envs(ale_py)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),

    torch.nn.Flatten(),
    torch.nn.Linear(288, 3)
).to(device)

model = torch.load("./pong_game/model.pt", weights_only=False)
model.eval()

env = gym.make('ALE/Pong-v5', render_mode="human")


done = False
episode_return = 0
obs, info = env.reset()
observations = []

while not done:
    env.render()

    if len(observations)<2:
        obs, reward, done, truncated, info = env.step(0)

        observations.append(obs)
        skip = False
    
    else:
        ot = observations[0][34:192, 80:145, 0]
        otp1 = observations[1][34:192, 80:145, 0]

        ot = ot/2+otp1
        ot[ot==216]=0.
        ot[ot!=0]=1.

   
        input = torch.tensor([[ot]]).float().to(device)

        pred = model(input)
        pred = torch.nn.functional.softmax(pred)
 
        action = torch.argmax(pred)

        if action>0:
            action += 1

        obs, reward, done, truncated, info = env.step(action)

        observations = observations[1:]
        observations.append(obs)
    
    time.sleep(0.1)

