import gymnasium as gym

env = gym.make(
    "CarRacing-v2",
    continuous=False
)

obs, _ = env.reset()

print(obs.shape)

env.close()