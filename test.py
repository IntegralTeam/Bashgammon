# Run testing process CNN on generated dataset
# For testing just run this file

import os
import torch

import torch.nn.functional as F
from model import CNN
from dataset import Dataset

# Global variables
BATCH = 3                       # BATCH - image in batch
model_save_path = "cnn_model"   # Directory for save file
model_name = "model_n1.pt"      # Name for file with saved model weights

def test(dataset):
    model = CNN()
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name)))
    # Choise GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        model.cuda(0)
    acc_val = 0.0
    acc_epoch = 100
    index = 0
    # Learning loop
    with torch.no_grad():
        # By games
        for gi in range(dataset.train_games_count()):
            # By epoch
            for epoch in range(2):
                # By images in game
                for i in range(dataset.train_image_count(gi, BATCH)):
                    # Parsed and load image,labels,steps from dataset
                    image_batch, label_batch, steps_batch = dataset.get_test_batch(gi, i, BATCH)
                    if device != "cpu":
                        image_batch = image_batch.to(device)
                        label_batch = label_batch.to(device)
                        steps_batch = steps_batch.to(device)
                    out = model(image_batch, steps_batch)

                    # Print accurancy
                    acc_val += 1 - F.mse_loss(out, label_batch)
                    index += 1
                    if index % acc_epoch == 0:
                        print("Acc", acc_val / acc_epoch)
                        acc_val = 0.0
    return

# Init dataset class and run testing process
dataset_ = Dataset("data",256,128)
test(dataset_)
