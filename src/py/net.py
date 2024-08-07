import torch.nn.functional as F

from torch import nn


class ResNet(nn.Module):
    def __init__(self, gameType, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(gameType.num_state_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, gameType.num_action_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(gameType.num_action_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(gameType.action_space_size, gameType.num_action_channels * gameType.nRows() * gameType.nCols()),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, gameType.num_state_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(gameType.num_state_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(gameType.state_space_size, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
