# DQN
# Austin Eng

# Environment
Install my requirements folder to environment
(Gymnasium / CarRacing-v2 setup)

# Replay Buffer
Allows you to store past driving experiences, so the agent learns from random past situations instead of only recent frames.

This breaks correlation between samples and makes DQN much more stable.

# CNN
The CNN takes in the current state (stacked frames) and learns features from images.

It answers: “what action should I take given this state?”

We use Q-learning here:

Q(s, a) = expected future reward for taking action a in state s

# Forward Pass Test
Input = (4 × 84 × 84) stacked frames

Flow:

Convolution layers extract spatial features
Fully connected layers process features
Output layer returns 5 Q-values (one per action)

Problem:
The network alone doesn’t know:

how to explore
how to train
how to update target networks
how to choose actions

#Epsilon-Greedy
Fixes exploration problem.

With probability ε → random action (explore)
Otherwise → best Q-value action (exploit)

Action mapping:

0 = do nothing
1 = steer right
2 = steer left
3 = gas
4 = brake

ε decays over time so the model shifts from exploring → exploiting.

# Optimizer
Needed in initialization.

Without it:

no gradient updates
no learning from loss
network cannot improve

Common: Adam optimizer

# Bellman Loss
Core of DQN learning.

Update rule:

Q_target = r + γ * max(Q(next_state))

Loss:

difference between predicted Q(s,a) and target Q-value

This pushes predictions toward better long-term reward estimation.

# Target Network
Separate network used to compute stable targets.

Why:

prevents constantly shifting Q-targets
stabilizes training
reduces divergence

Updated every fixed number of steps from main network.

# Training Loop
Main cycle:
get current state (stacked frames)
choose action (epsilon-greedy)
step environment
get reward + next state
store transition in replay buffer
sample random batch
compute Bellman targets
calculate loss
backprop update network
occasionally update target network

# Frame Stacking
We use multiple frames (usually 4) to give motion info.

This helps the agent understand:

speed
direction
movement trends

Without stacking → model only sees single static image.

# Reward Signal
The reward guides learning.

In CarRacing:

staying on track = positive reward
going off track = penalty
slow progress = low reward

The agent learns to maximize total episode reward.

# Exploration vs Exploitation Tradeoff
Early training:
mostly exploration (random actions)

Later training:

mostly exploitation (greedy Q-values)

This balance is critical for convergence.

# Model Intuition
The network is basically learning:

“If I see this road pattern, what action gives me the highest long-term score?”