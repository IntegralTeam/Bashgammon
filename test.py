import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CNN
from dataset import Dataset

model_save_path = "models"
model_name = "model_n0.pt"
BATCH = 2

def test(dataset):
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        model.cuda(0)
    acc_val = 0.0
    acc_epoch = 100
    index = 0
    with torch.no_grad():
        for gi in range(dataset.train_games_count()):
            for epoch in range(2):
                for i in range(dataset.train_image_count(gi, BATCH)):
                    image_batch, label_batch, steps_batch = dataset.get_test_batch(gi, i, BATCH)
                    if device != "cpu":
                        image_batch = image_batch.to(device)
                        label_batch = label_batch.to(device)
                        steps_batch = steps_batch.to(device)
                    for s in steps_batch:
                        out = model(image_batch, s)
                        index += 1
                        acc_val += (label_batch - out)
                        if index % acc_epoch == 0:
                            print("Loss", acc_val / acc_epoch)
                            acc_val = 0.0
    return

dataset_ = Dataset("data",256,128)
test(dataset_)
