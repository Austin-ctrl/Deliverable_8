from replay_buffer import replay_buffer
import torch

buffer = replay_buffer(100)

state = torch.randn(4,84,84)

buffer.add(
    state,
    3,
    1.0,
    state,
    False
)

print(len(buffer))

batch = buffer.sample(1)

print(batch)