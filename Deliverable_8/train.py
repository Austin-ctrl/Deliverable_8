import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import gymnasium as gym
 
from model import ActorCritic
from worker import Worker
from optimizer import SharedRMSProp
from plot import plot_results
from config import ENV_NAME, NUM_WORKERS, MAX_EPISODES, LR, SEED
 
 
def train():
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(SEED)
 
    # Probe environment dimensions
    env = gym.make(ENV_NAME)
    obs_dim    = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole
    env.close()
 
    print(f"Environment : {ENV_NAME}")
    print(f"Obs dim : {obs_dim} | Action dim : {action_dim}")
    print(f"Workers : {NUM_WORKERS}  |  Max episodes : {MAX_EPISODES:,}")
    print("─" * 50)
 
    # Shared global model + optimizer
    global_model = ActorCritic(obs_dim, action_dim)
    global_model.share_memory()
    optimizer = SharedRMSProp(global_model.parameters(), lr=LR)
 
    # Shared state across workers
    global_ep_counter = mp.Value('i', 0)
    reward_queue  = mp.Queue()
 
    # Launch workers
    workers = [
        Worker(
            worker_id=i,
            global_model=global_model,
            optimizer=optimizer,
            global_ep_counter=global_ep_counter,
            reward_queue=reward_queue,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()
 
    # Collect results from the reward queue
    all_episodes, all_rewards = [], []
    done_workers = 0
    start_time   = time.time()
 
    while done_workers < NUM_WORKERS:
        item = reward_queue.get()
        if item is None:
            done_workers += 1
            continue
        episode, reward = item
        all_episodes.append(episode)
        all_rewards.append(reward)
 
        if len(all_rewards) % 100 == 0:
            avg = np.mean(all_rewards[-100:])
            elapsed = time.time() - start_time
            print(f"  ep {episode:>5,} | last-100 avg: {avg:6.1f} | {elapsed:.0f}s elapsed")
 
    for w in workers:
        w.join()
 
    elapsed = time.time() - start_time
    print("─" * 50)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Total episodes : {len(all_rewards)}")
    print(f"Final 100-ep avg : {np.mean(all_rewards[-100:]):.1f}")
 
    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    torch.save(global_model.state_dict(), "outputs/a3c_cartpole.pt")
    print("Model saved → outputs/a3c_cartpole.pt")
 
    plot_results(all_episodes, all_rewards, save_path="outputs/training_results.png")
 
 
if __name__ == "__main__":
    train()