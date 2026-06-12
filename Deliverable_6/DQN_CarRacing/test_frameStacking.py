import gymnasium as gym
from preprocess import preprocess_frame
from collections import deque
import torch

env = gym.make(
    "CarRacing-v2",
    continuous=False
)

obs, _ = env.reset()

frame = preprocess_frame(obs)

frames = deque(maxlen=4)

for _ in range(4):
    frames.append(frame)

state = torch.stack(list(frames))

print(state.shape)

env.close()