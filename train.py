# Run training CNN on generated dataset
# For training just run this file

import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from dataset import Dataset

# Global variables
BATCH = 3                       # BATCH - image in batch
model_save_path = "cnn_model"   # Directory for save file
model_name = "model_n1.pt"      # Name for file with saved model weights


def train(dataset):
    model = CNN()
    # Choise GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        model.cuda(0)

    # Loss function for evaluate error
    criterion = nn.MSELoss()
    # Optimization method
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    loss_val = 0.0
    loss_epoch = 100
    index = 0
    # Learning loop
    # By games
    for gi in range(dataset.train_games_count()):
        # By epoch
        for epoch in range(3):
            # By images in game
            for i in range(dataset.train_image_count(gi, BATCH)):
                # Parsed and load image,labels,steps from dataset
                image_batch, label_batch, steps_batch = dataset.get_train_batch(gi, i, BATCH)
                if device != "cpu":
                    image_batch = image_batch.to(device)
                    label_batch = label_batch.to(device)
                    steps_batch = steps_batch.to(device)

                # Forward and learning
                optimizer.zero_grad()
                out = model(image_batch, steps_batch)
                loss = criterion(out, label_batch)
                loss.backward()
                optimizer.step()

                # Print loss
                index += 1
                loss_val += loss
                if index % loss_epoch == 0:
                    print("Loss", loss_val / loss_epoch)
                    loss_val = 0.0
    # Save model weights
    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    return

# Init dataset class and run training process
dataset_ = Dataset("data",256,128)
train(dataset_)


