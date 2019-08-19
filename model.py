import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NET_CHAIN = [8,32]

# CNN architecture class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layers initialization
        self.conv1 = nn.Conv2d(1,NET_CHAIN[0], 5,1,2)
        self.bn1 = nn.BatchNorm2d(NET_CHAIN[0])

        self.conv2 = nn.Conv2d(NET_CHAIN[0], NET_CHAIN[1], 5,1,2)
        self.bn2 = nn.BatchNorm2d(NET_CHAIN[1])

        self.relu = nn.ReLU()
        self.pool4 = nn.MaxPool2d(4)

        self.fc1 = nn.Linear(NET_CHAIN[-1] * 8 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 63)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 48)
        self.fc6 = nn.Linear(48, 25)

    def forward(self, input, step):
        # Convolution -> Relu -> pool2d
        x = self.relu(self.conv1(input))
        x = self.pool4(x)
        x = self.relu(self.conv2(x))
        x = self.pool4(x)

        # Linear -> relu ->
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Concatenate output from "fc3" with "step"
        x = torch.cat((x, step), dim=1)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x