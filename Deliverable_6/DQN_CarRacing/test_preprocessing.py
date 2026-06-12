import gymnasium as gym
from preprocess import preprocess_frame

env = gym.make(
    "CarRacing-v2",
    continuous=False
)

obs, _ = env.reset()

frame = preprocess_frame(obs)

print(frame.shape)

env.close()