from agent import Agent
from replay_buffer import replay_buffer
import torch

agent = Agent()

buffer = replay_buffer(100)

for _ in range(64):

    state = torch.randn(4,84,84)

    next_state = torch.randn(4,84,84)

    buffer.add(
        state,
        2,
        1.0,
        next_state,
        False
    )

agent.train_step(
    buffer,
    32
)

print("Training step completed")