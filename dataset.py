import os
import numpy as np
from PIL import Image
import pandas as pd
import json
import torch

class Dataset():
    def __init__(self, path_to_root, width, height, train_to_test=75.0):
        self.root = path_to_root
        self.img_width = width
        self.image_height = height
        if train_to_test > 100 or train_to_test <= 0:
            train_to_test = 75
        f = open(os.path.join(self.root, "result.txt"), "r")
        lines = f.readlines()[1:-1]

        self.game_result = {}
        self.images = {}
        self.labels = {}
        self.steps = {}
        for x,l in enumerate(lines):
            spl = l[:-1].split(",")
            game = spl[0]
            winner = True if spl[1] == "True" else False

            game_dir = os.path.join(self.root, game)
            images = []
            labels = []
            steps = []
            moves = pd.read_csv(os.path.join(game_dir, "moves.csv"))
            for n in range(moves.shape[0]):
                moves_list = self.__strin_to_array(moves["AvailableMoves"][n])
                images.append(os.path.join(game_dir, str(n) + ".png"))
                steps.append(self.__normalize_steps(moves_list))
                labels.append(self.__normalize_labels(moves["Player"].iloc[n]))
            self.images[x] = images
            self.labels[x] = labels
            self.steps[x] = steps
            self.game_result[x] = winner

        self.train_to_test = train_to_test / 100.0
        self.train_game_c = int(len(self.game_result) * self.train_to_test)
        self.test_game_c = int(len(self.game_result) * (1 - self.train_to_test))
        return

    def train_image_count(self, game_id, step=1):
        if game_id >= self.train_game_c:
            return 0
        return len(self.images[game_id]) // step

    def test_image_count(self, game_id, step=1):
        if game_id >= self.test_game_c:
            return 0
        return len(self.images[game_id]) // step

    def train_games_count(self):
        return self.train_game_c

    def test_games_count(self):
        return self.test_game_c

    def __strin_to_array(self, s):
        s = s.replace('True', '1').replace('False', '0')
        out = np.array(json.loads(s))
        return np.delete(out, [1,3], axis=1)

    def __normalize_labels(self, player):
        if player == True:
            return np.array([1.0])
        else:
            return np.array([0.0])

    def __normalize_steps(self, step):
        step[:,0] = (step[:,0]+1) / 24
        step[:, 1] = (step[:, 1] - 1) / 5
        return step

    def __get_train_image(self, game_id, i, batch_sz, mode=0):
        color_chanell = 1 if mode == 0 else 3
        slice = np.s_[batch_sz * i: batch_sz * (i + 1)]
        out = np.ndarray(shape=(batch_sz, color_chanell, self.img_width, self.image_height))
        for img_id, name in enumerate(self.images[game_id][slice]):
            out[img_id, 0] = np.array(
                Image.open(name).resize((self.img_width, self.image_height))
            ).transpose((1,0))
        return out

    def __get_train_label(self, game_id, i, batch_sz):
        slice = np.s_[batch_sz * i: batch_sz * (i + 1)]
        out = np.ndarray(shape=(batch_sz, 1))
        for id, label in enumerate(self.labels[game_id][slice]):
            out[id] = label
        return out

    def __get_train_steps(self, game_id, i, batch_sz):
        slice = np.s_[batch_sz * i: batch_sz * (i + 1)]
        step_ls = self.steps[game_id][slice]
        _min = 10000
        for s in step_ls:
            _min = len(s) if len(s) < _min else _min
        for si, s in enumerate(step_ls):
            step_ls[si] = s[:_min]
        return np.array(step_ls).transpose((1, 0, 2))


    def get_train_batch(self, game_id, i, batch_sz):
        return torch.tensor(self.__get_train_image(game_id, i, batch_sz)).float(),\
               torch.tensor(self.__get_train_label(game_id, i, batch_sz)).float(), \
               torch.tensor(self.__get_train_steps(game_id, i, batch_sz)).float()

    def get_test_batch(self, game_id, i, batch_sz):
        return torch.tensor(self.__get_train_image(game_id, i, batch_sz)).float(), \
               torch.tensor(self.__get_train_label(game_id, i, batch_sz)).float(), \
               torch.tensor(self.__get_train_steps(game_id, i, batch_sz)).float()