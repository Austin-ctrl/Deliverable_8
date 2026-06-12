import csv
import os
import gymnasium as gym
import torch
import numpy as np
from collections import deque
from agent import Agent
from replay_buffer import replay_buffer
from preprocess import preprocess_frame
os.makedirs("logs", exist_ok=True)
log_file = open("logs/training_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["episode", "reward", "loss", "epsilon", "total_steps"])
# train.py


# Create environment
env = gym.make(
    "CarRacing-v2",
    continuous=False
)

# Create agent from the agent class
agent = Agent()

# Create replay buffer
buffer = replay_buffer(100000)

# Hyperparameters
num_episodes = 200
batch_size = 32
target_update_freq = 1000

total_steps = 0

for episode in range(num_episodes):

    # Reset environment
    obs, _ = env.reset()

    # Preprocess first frame
    frame = preprocess_frame(obs)

    # Create initial 4-frame stack
    frames = deque(maxlen=4)

    for _ in range(4):
        frames.append(frame)

    state = torch.stack(list(frames))

    done = False
    episode_reward = 0
    episode_losses = []
    while not done:

        # Add batch dimension
        state_input = state.unsqueeze(0)

        # Agent chooses action
        action = agent.select_action(state_input)

        # Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Process next frame
        next_frame = preprocess_frame(next_obs)

        # Update frame stack
        frames.append(next_frame)

        next_state = torch.stack(list(frames))

        # Store experience
        buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn
        
        if len(buffer) > batch_size:
            loss = agent.train_step(
                buffer,
                batch_size
            )
            episode_losses.append(loss)
            agent.decay_epsilon()

        # Update target network
        if total_steps % target_update_freq == 0:
            agent.update_target_net()

        state = next_state

        episode_reward += reward

        total_steps += 1
        if total_steps % 200 == 0:
            print(f"  Step {total_steps} | Buffer {len(buffer)} | ε {agent.epsilon:.4f}")

    print(
        f"Episode {episode} | Reward: {episode_reward:.2f}"
    )
    writer.writerow([episode, episode_reward, np.mean(episode_losses), agent.epsilon, total_steps])
    log_file.flush()

env.close()
log_file.close()
