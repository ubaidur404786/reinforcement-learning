import numpy as np
import gymnasium as gym
import os
import pygame
import ale_py 
import time

# Ensure the directory exists
os.makedirs("./pong_game/data", exist_ok=True)

# 1. Setup Environment
env = gym.make('ALE/Pong-v5', render_mode="human") # Set to human to see the window
obs, info = env.reset()

# 2. Setup Pygame for mouse/clock
pygame.init()
clock = pygame.time.Clock()

buffered = []
cpt = 0
FPS = 15 # SLOW SPEED: Increase to 30 or 60 for faster gameplay

print(f"--- CONTROLS ---")
print(f"1. Mouse Wheel: Scroll UP/DOWN to move paddle")
print(f"2. Arrows: UP/DOWN keys also work")
print(f"3. Speed: {FPS} FPS (Slow/Natural)")

running = True
while running:
    action = 0 # Default: Stay still

    # 3. Handle Events (Keyboard + Mouse Wheel)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Mouse Wheel Detection
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: action = 2 # Scroll Up
            if event.button == 5: action = 3 # Scroll Down

    # Keyboard Backup
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]: action = 2
    if keys[pygame.K_DOWN]: action = 3

    # 4. Step Environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # 5. Data Collection (Your original logic)
    # Note: obs is the RGB image
    ot = obs[34:192, 80:145, 0]
    otp1 = next_obs[34:192, 80:145, 0]

    buffered.append({"name": f"{cpt}_{action}.npy", "frames": np.stack([ot, otp1], axis=0)})
    cpt += 1

    # Ball hit detection logic
    for p in range(len(otp1)):
        if (otp1[p, 54] > 150 or otp1[p, 57] > 150) and (otp1[p, 60] < 95 or otp1[p, 61] < 95 or otp1[p, 62] < 95):
            to_save = buffered[-10:]
            for f in to_save:
                np.save(f"./pong_game/data/{f['name']}", f["frames"])
            buffered = []
            break

    obs = next_obs

    if terminated or truncated:
        obs, info = env.reset()

    # Control Game Speed
    clock.tick(FPS)

pygame.quit()
env.close()