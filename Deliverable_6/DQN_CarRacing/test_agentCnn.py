from agent import Agent
import torch

agent = Agent()

state = torch.randn(
    1,
    4,
    84,
    84
)

action = agent.select_action(state)

print(action)