import os

import torch
from torchvision.io.image import read_image

import matplotlib.pyplot as plt


if __name__ == '__main__':
    segmentation_dir = ''
    mask_save_dir = ''

    # for seg in os.listdir(segmentation_dir):
    #     seg_path = os.path.join(segmentation_dir, seg)
    #     count = os.path.basename(seg_path).split('_')[1]
    #     seg_img = read_image(seg_path)

    # 先读一张图片看看阈值
