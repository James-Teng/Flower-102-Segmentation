import sys
import time
import os
import json
from json.decoder import JSONDecodeError
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from collections.abc import Callable


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossSaver:
    def __init__(self):
        self.loss_list = []

    def reset(self):
        self.loss_list = []

    def append(self, loss):
        self.loss_list.append(loss)

    def to_np_array(self):
        return np.array(self.loss_list)

    def save_to_file(self, file_path):
        np_loss_list = np.array(self.loss_list)
        np.save(file_path, np_loss_list)


def load_loss_file(file_path):
    data = np.load(file_path)
    return data

