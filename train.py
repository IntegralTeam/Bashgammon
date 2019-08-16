import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CNN
from dataset import Dataset

BATCH = 3
model_save_path = "models"
model_name = "model_n0.pt"


def train(dataset):
    model = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        model.cuda(0)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss_val = 0.0
    loss_epoch = 100
    index = 0
    for gi in range(dataset.train_games_count()):
        for epoch in range(2):
            for i in range(dataset.train_image_count(gi, BATCH)):
                image_batch, label_batch, steps_batch = dataset.get_train_batch(gi, i, BATCH)
                if device != "cpu":
                    image_batch = image_batch.to(device)
                    label_batch = label_batch.to(device)
                    steps_batch = steps_batch.to(device)
                for s in steps_batch:
                    optimizer.zero_grad()
                    out = model(image_batch, s)
                    loss = criterion(out, label_batch)
                    loss.backward()
                    optimizer.step()
                    index += 1
                    loss_val += loss
                    if index % loss_epoch == 0:
                        print("Loss", loss_val / loss_epoch)
                        loss_val = 0.0
    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    return

dataset_ = Dataset("data",256,128)
train(dataset_)


