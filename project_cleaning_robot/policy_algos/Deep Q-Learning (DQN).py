import numpy as np
import random
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# COLORS (same as your Q-learning)
# =========================================================
RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"
BROWN  = "\033[33m"


# =========================================================
# ENVIRONMENT
# =========================================================
class CleaningRoom:

    def __init__(self):
        self.rows = 10
        self.cols = 10
        self.start_pos = (0, 0)

        self.walls = {
            (1,0),(1,1),(2,0),(2,1),
            (4,0),(4,1),(4,2),(4,3),
            (5,0),(5,1),(5,2),(5,3),
            (6,0),(6,1),(6,2),(6,3),
            (4,6),(4,7),(4,8),(4,9),
            (5,6),(5,7),(5,8),(5,9)
        }

        self.cleanable = {
            (r,c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r,c) not in self.walls
        }

        self.reset()

    def reset(self):
        self.robot_pos = self.start_pos
        self.cleaned_tiles = set([self.start_pos])
        self.visit_count = np.zeros((self.rows, self.cols))
        self.last_pos = None
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.rows, self.cols), dtype=np.float32)

        for (r,c) in self.cleaned_tiles:
            state[r,c] = 0.5

        for (r,c) in self.walls:
            state[r,c] = -1.0

        x,y = self.robot_pos
        state[x,y] = 1.0

        return state.flatten()

    def step(self, action):
        x,y = self.robot_pos

        if action == 0: x -= 1
        elif action == 1: y += 1
        elif action == 2: x += 1
        elif action == 3: y -= 1

        new_pos = (x,y)

        if (
            x < 0 or x >= self.rows or
            y < 0 or y >= self.cols or
            new_pos in self.walls
        ):
            return self.get_state(), -20, False

        reward = -1

        if new_pos not in self.cleaned_tiles:
            reward += 20
            self.cleaned_tiles.add(new_pos)
        else:
            reward -= 10

        if new_pos == self.last_pos:
            reward -= 30

        self.visit_count[new_pos] += 1
        reward -= self.visit_count[new_pos]

        if self.cleaned_tiles == self.cleanable:
            old_d = abs(self.robot_pos[0]) + abs(self.robot_pos[1])
            new_d = abs(x) + abs(y)

            reward += 15 if new_d < old_d else -15

            if new_pos == self.start_pos:
                reward += 100
                self.robot_pos = new_pos
                return self.get_state(), reward, True

        self.last_pos = self.robot_pos
        self.robot_pos = new_pos
        return self.get_state(), reward, False


# =========================================================
# PRINT ROOM (SAME STYLE AS YOUR Q-LEARNING)
# =========================================================
def print_room(env, ep, step, eps):
    print("\033[H\033[J", end="")
    print(f"Episode {ep} | Step {step} | ε={eps:.2f}")
    print("="*50)

    for i in range(env.rows):
        row = ""
        for j in range(env.cols):
            p = (i,j)
            if p == env.robot_pos:
                row += RED + " 🤖 " + RESET
            elif p == env.start_pos:
                row += GREEN + " 🟩 " + RESET
            elif p in env.walls:
                row += GRAY + " ⬛ " + RESET
            elif p in env.cleaned_tiles:
                row += WHITE + " ⬜ " + RESET
            else:
                row += BROWN + " 🟫 " + RESET
        print(row)

    print("="*50)
    print(f"Cleaned: {len(env.cleaned_tiles)} / {len(env.cleanable)}")


# =========================================================
# DQN NETWORK
# =========================================================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# DQN AGENT
# =========================================================
class DQNAgent:

    def __init__(self):
        self.policy = DQN()
        self.target = DQN()
        self.target.load_state_dict(self.policy.state_dict())

        self.optim = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.memory = deque(maxlen=50000)

        self.gamma = 0.95
        self.batch_size = 64

        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.min_eps = 0.05

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,3)

        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policy(s)).item()

    def store(self, s,a,r,ns,d):
        self.memory.append((s,a,r,ns,d))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s,a,r,ns,d = zip(*batch)

        s = torch.FloatTensor(s)
        ns = torch.FloatTensor(ns)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        d = torch.FloatTensor(d)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target(ns).max(1)[0]
        target = r + self.gamma * next_q * (1-d)

        loss = nn.MSELoss()(q, target.detach())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())


# =========================================================
# TRAINING LOOP (WITH VISUAL OUTPUT)
# =========================================================
env = CleaningRoom()
agent = DQNAgent()

EPISODES = 400
TARGET_UPDATE = 10
success = 0

for ep in range(1, EPISODES+1):
    state = env.reset()

    for step in range(3000):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.store(state, action, reward, next_state, done)
        agent.train()

        print_room(env, ep, step, agent.epsilon)
        time.sleep(0.01)

        state = next_state

        if done:
            success += 1
            print("\nSUCCESS: cleaned all tiles and returned home")
            break

    agent.epsilon = max(agent.min_eps, agent.epsilon * agent.eps_decay)

    if ep % TARGET_UPDATE == 0:
        agent.update_target()

    if success >= 10 and agent.epsilon <= 0.06:
        print("\nOPTIMAL POLICY LEARNED (DQN)")
        break

print("\nTraining finished.")
