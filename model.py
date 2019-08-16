import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NET_CHAIN = [8,32]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,NET_CHAIN[0], 5,1,2)
        self.bn1 = nn.BatchNorm2d(NET_CHAIN[0])

        self.conv2 = nn.Conv2d(NET_CHAIN[0], NET_CHAIN[1], 5,1,2)
        self.bn2 = nn.BatchNorm2d(NET_CHAIN[1])

        #self.conv3 = nn.Conv2d(NET_CHAIN[1], NET_CHAIN[2],5,1,2)
        #self.bn3 = nn.BatchNorm2d(NET_CHAIN[2])

        #self.conv4 = nn.Conv2d(NET_CHAIN[2], NET_CHAIN[3], 5,1,2)

        self.relu = nn.ReLU()
        self.pool4 = nn.MaxPool2d(4)
        #self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(NET_CHAIN[-1] * 8 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 62)
        #self.fc4 = nn.Linear(256, 62)

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, input, step):
        x = self.relu(self.conv1(input))
        x = self.pool4(x)
        x = self.relu(self.conv2(x))
        x = self.pool4(x)
        #x = self.relu(self.conv3(x))
        #x = self.pool2d(x)
        #x = self.relu(self.conv4(x))
        #x = self.pool2d(x)
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        x = torch.cat((x, step), dim=1)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x