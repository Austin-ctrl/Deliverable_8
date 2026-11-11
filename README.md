# A3C Reinforcement Learning — Week 8 Deliverable

## Overview

This project focuses on improving the performance of an **Asynchronous Advantage Actor-Critic (A3C)** reinforcement learning model through hyperparameter tuning and neural network architecture adjustments.

The goal was to improve training performance and stability by experimenting with learning rates, training episodes, and the number of neural network layers.

Through multiple iterations, I improved the model's performance compared to the initial implementation while exploring the tradeoffs between model complexity, training time, and learning stability.

## Technologies Used

- **Python** — Model implementation and experimentation
- **A3C (Asynchronous Advantage Actor-Critic)** — Reinforcement learning algorithm
- **Neural Networks** — Policy and value function learning
- **Hyperparameter Tuning** — Learning rate, training duration, and network architecture

## Model Improvements

The initial A3C model demonstrated relatively poor performance, motivating several changes to the training configuration.

### 1. Increasing Training Episodes

The initial configuration used 1,500 training episodes.

I experimented with increasing the training duration to 3,000 episodes to give the agent more opportunities to learn from its environment.

Longer training was intended to improve the model's performance by allowing additional experience collection and parameter updates.

### 2. Learning Rate Optimization

I experimented with multiple learning rates to improve training stability.

The learning rate was adjusted to `3e-4` and later reduced to `2e-4`.

The final learning rate of `2e-4` produced more stable learning behavior during experimentation while still allowing the model to improve.

### 3. Neural Network Architecture

I experimented with increasing the number of neural network layers.

- **Two-layer architecture:** Used during initial model improvements.
- **Three-layer architecture:** Tested to increase model capacity.
- **256 neurons:** Explored as a larger network configuration.

Initially, adding a third layer resulted in worse performance. Adjusting the number of neurons helped explore whether increased model capacity could improve learning.

However, larger networks also significantly increased training time, highlighting the tradeoff between model complexity and computational cost.

## Experiment Summary

| Parameter | Initial / Earlier Configuration | Later Experiment |
|---|---|---|
| Training episodes | 1,500 | 3,000 tested |
| Learning rate | `3e-4` | `2e-4` |
| Hidden layers | 2 | 3 |
| Neurons | Earlier configuration | 256 explored |
| Performance | Lower initial results | Improved after tuning |

*Note: The final training episode count and exact performance metrics should be confirmed from the experiment logs.*

## Results

After adjusting the model's hyperparameters, I observed improved performance compared to the previous deliverable.

Key observations included:

- Reducing the learning rate to `2e-4` helped stabilize training.
- Increasing the number of training episodes provided more opportunities for the agent to learn.
- Increasing the number of neural network layers did not automatically improve performance.
- Larger network configurations increased computational cost and training time.
- Hyperparameter tuning required balancing learning stability, model complexity, and available training time.

These experiments demonstrated that improving a reinforcement learning model involves more than simply increasing its size.

## Challenges Encountered

### Training Stability

The initial A3C model performed poorly. Finding a suitable learning rate required experimentation because overly aggressive parameter updates could make learning unstable.

### Model Complexity

Adding a third neural network layer initially reduced performance.

This showed that deeper architectures do not necessarily produce better results without appropriate tuning.

### Computational Cost

Increasing the number of neurons to 256 significantly increased training time, making experimentation more expensive.

## Future Improvements

Potential next steps include:

1. **Increase training duration:** Experiment with 5,000 episodes to determine whether additional training improves performance.
2. **Compare network architectures:** Evaluate two-layer and three-layer networks under consistent training conditions.
3. **Perform systematic hyperparameter tuning:** Compare learning rates and hidden-layer sizes while keeping other parameters fixed.
4. **Track performance metrics:** Record episode rewards, moving-average rewards, and training duration to evaluate improvements more precisely.
5. **Evaluate reproducibility:** Repeat experiments with different random seeds to determine whether improvements are consistent.

## Key Takeaways

This project strengthened my understanding of reinforcement learning, neural network architecture, and hyperparameter optimization.

I gained practical experience diagnosing poor model performance, experimenting with learning rates, and evaluating the relationship between model complexity and training efficiency.

Most importantly, the project demonstrated the importance of iterative experimentation and measuring performance rather than assuming that larger networks or longer training will always produce better results.

## Repository

**GitHub:** https://github.com/Austin-ctrl/Deliverable_8

This repository contains the Week 8 A3C reinforcement learning deliverable and associated implementation.
