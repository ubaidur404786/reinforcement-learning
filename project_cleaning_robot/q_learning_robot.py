import numpy as np
import random
import time

# colors only to show grid nicely in terminal
RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"
BROWN  = "\033[33m"


# this class is the environment (room)
# robot learns only by rewards, no map knowledge
class CleaningRoom:

    def __init__(self):
        # grid size
        self.rows = 10
        self.cols = 10

        # start position (also end position)
        self.start_pos = (0, 0)

        # walls = places robot cannot go
        self.walls = {
            (1,0),(1,1),(2,0),(2,1),
            (4,0),(4,1),(4,2),(4,3),
            (5,0),(5,1),(5,2),(5,3),
            (6,0),(6,1),(6,2),(6,3),
            (4,6),(4,7),(4,8),(4,9),
            (5,6),(5,7),(5,8),(5,9)
        }

        # all tiles that are not walls can be cleaned
        self.cleanable = {
            (r,c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r,c) not in self.walls
        }

        self.reset()


    # reset environment for new episode
    def reset(self):
        self.robot_pos = self.start_pos  # robot starts at home

        self.cleaned_tiles = set([self.start_pos])  # start tile is clean

        self.visit_count = np.zeros((self.rows, self.cols))  
        # count how many times robot visits a tile (to avoid loops)

        self.last_pos = None  # remember last position (to stop back-forth)

        return self.robot_pos


    # one step in environment
    def step(self, action):
        x, y = self.robot_pos

        # action to movement
        if action == 0: x -= 1
        elif action == 1: y += 1
        elif action == 2: x += 1
        elif action == 3: y -= 1

        new_pos = (x, y)

        # hit wall or outside grid
        if (
            x < 0 or x >= self.rows or
            y < 0 or y >= self.cols or
            new_pos in self.walls
        ):
            return self.robot_pos, -20

        reward = -1  # step cost

        # clean new tile
        if new_pos not in self.cleaned_tiles:
            reward += 20
            self.cleaned_tiles.add(new_pos)
        else:
            reward -= 10  # reclean is bad

        # going back to last position = loop
        if new_pos == self.last_pos:
            reward -= 30

        # too many visits = loop
        self.visit_count[new_pos] += 1
        reward -= self.visit_count[new_pos]

        #   AFTER ALL TILES CLEANED
        if self.all_cleaned():
            old_dist = abs(self.robot_pos[0] - self.start_pos[0]) + \
                    abs(self.robot_pos[1] - self.start_pos[1])

            new_dist = abs(new_pos[0] - self.start_pos[0]) + \
                    abs(new_pos[1] - self.start_pos[1])

            # moving toward home is good
            if new_dist < old_dist:
                reward += 15
            else:
                reward -= 15

            # reaching home ends task
            if new_pos == self.start_pos:
                reward += 100  # big success reward

        self.last_pos = self.robot_pos
        self.robot_pos = new_pos

        return new_pos, reward



    # check if all tiles are cleaned
    def all_cleaned(self):
        return self.cleaned_tiles == self.cleanable


# Q-learning agent (brain of robot)
class QLearningAgent:

    def __init__(self):
        # Q-table: [row][col][action]
        self.q = np.zeros((10,10,4))

        self.lr = 0.1        # learning rate
        self.gamma = 0.95   # future reward importance

        self.epsilon = 0.1  # exploration rate
        self.eps_decay = 0.995
        self.min_eps = 0.05


    # choose action using epsilon-greedy
    def choose_action(self, state):
        x, y = state

        # explore: try random action
        if random.random() < self.epsilon:
            return random.randint(0,3)

        # exploit: choose best known action
        return np.argmax(self.q[x,y])


    # Q-learning update rule
    def learn(self, s, a, r, ns):
        x,y = s
        nx,ny = ns

        # Q(s,a) = Q(s,a) + lr * (reward + gamma*max(Q(next)) - Q(s,a))
        self.q[x,y,a] += self.lr * (
            r + self.gamma * np.max(self.q[nx,ny])
            - self.q[x,y,a]
        )


# print the room on screen
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


# ================= TRAINING =================
env = CleaningRoom()
agent = QLearningAgent()

EPISODES = 300
MAX_STEPS = 2500
successful_runs = 0

for ep in range(1, EPISODES+1):
    state = env.reset()

    for step in range(MAX_STEPS):
        action = agent.choose_action(state)      # choose action
        next_state, reward = env.step(action)    # take action
        agent.learn(state, action, reward, next_state)  # learn from it

        print_room(env, ep, step, agent.epsilon)
        time.sleep(0.01)

        state = next_state

        # episode ends only when:
        # all tiles are clean AND robot is back home
        if env.all_cleaned() and env.robot_pos == env.start_pos:
            successful_runs += 1
            print("\nSUCCESS: cleaned all tiles and returned home")
            break

    # slowly reduce exploration
    agent.epsilon = max(agent.min_eps, agent.epsilon * agent.eps_decay)

    # if robot succeeds many times → policy is stable
    if successful_runs >= 10 and agent.epsilon <= 0.06:
        print("\nOPTIMAL POLICY FOUND")
        break

print("\nTraining finished.")
