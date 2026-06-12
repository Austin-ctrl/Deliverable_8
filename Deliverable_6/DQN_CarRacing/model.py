import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        # convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4),  # input: 4x96x96, output: 32x22x22 (looks for road edges, grass, car body and curves)
            nn.ReLU(),

            nn.Conv2d(32,64,4,stride=2),  # input: 32x22x22, output: 64x10x10 (looks for more)
            nn.ReLU(),

            nn.Conv2d(64,64,3,stride=1),  # input: 64x10x10, output: 64x8x8 (looks for more)
            nn.ReLU()
        ) 
        # convolution layers let us see
        # Road curving left, Car near the grass, speed i fast, and an incoming turn
        
        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)  # input: 512, output: n_actions (looks for the best action to take)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
        # quick test:
if __name__ == "__main__":

    net = DQN(5)

    x = torch.randn(1, 4, 84, 84)

    y = net(x)

    print("Output shape:", y.shape)
    print(y)