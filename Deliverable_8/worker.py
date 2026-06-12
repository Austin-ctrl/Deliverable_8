import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical

from model import ActorCritic
from config import (
    ENV_NAME, T_MAX, GAMMA, ENTROPY_BETA,
    VALUE_LOSS_COEF, GRAD_CLIP, MAX_EPISODES, SEED
)


class Worker(mp.Process):

    def __init__(self, worker_id, global_model, optimizer,
                 global_ep_counter, reward_queue, obs_dim, action_dim):
        super().__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.global_ep_counter = global_ep_counter
        self.reward_queue = reward_queue
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def run(self):
        torch.manual_seed(SEED + self.worker_id)
        local_model = ActorCritic(self.obs_dim, self.action_dim)
        env = gym.make(ENV_NAME)

        while True:
            with self.global_ep_counter.get_lock():
                if self.global_ep_counter.value >= MAX_EPISODES:
                    break
                self.global_ep_counter.value += 1
                episode_num = self.global_ep_counter.value

            # sync with global model
            local_model.load_state_dict(self.global_model.state_dict())

            obs, _ = env.reset(seed=SEED + episode_num)
            ep_reward = 0
            done = False

            while not done:
                rollout_obs, rollout_rewards = [], []
                rollout_log_probs, rollout_values = [], []

                for _ in range(T_MAX):
                    action, log_prob, value = local_model.get_action(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    rollout_obs.append(obs)
                    rollout_rewards.append(reward)
                    rollout_log_probs.append(log_prob)
                    rollout_values.append(value)

                    ep_reward += reward
                    obs = next_obs

                    if done:
                        break

                if not done:
                    with torch.no_grad():
                        x = torch.FloatTensor(obs).unsqueeze(0)
                        _, R = local_model(x)
                        R = R.item()
                else:
                    R = 0.0

                policy_loss = torch.tensor(0.0)
                value_loss = torch.tensor(0.0)
                entropy_loss = torch.tensor(0.0)

                for i in reversed(range(len(rollout_rewards))):
                    R = rollout_rewards[i] + GAMMA * R
                    advantage = R - rollout_values[i].item()

                    policy_loss = policy_loss - rollout_log_probs[i] * advantage
                    value_loss = value_loss + VALUE_LOSS_COEF * (R - rollout_values[i]) ** 2

                    x = torch.FloatTensor(rollout_obs[i]).unsqueeze(0)
                    logits, _ = local_model(x)
                    dist = Categorical(logits=logits)
                    entropy_loss = entropy_loss - ENTROPY_BETA * dist.entropy()

                total_loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_CLIP)

                for local_p, global_p in zip(local_model.parameters(),
                                             self.global_model.parameters()):
                    if global_p.grad is None:
                        global_p._grad = local_p.grad.clone()
                    else:
                        global_p._grad += local_p.grad

                self.optimizer.step()

            self.reward_queue.put((episode_num, ep_reward))

        env.close()
        self.reward_queue.put(None)