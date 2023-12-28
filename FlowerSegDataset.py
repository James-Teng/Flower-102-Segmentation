import os
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision
from torchvision.io.image import read_image

import matplotlib.pyplot as plt


# todo 分割数据集吗
class Flower102Seg(Dataset):
    def __init__(
            self,
            img_root: str,
            target_root: str,
            crop_size: int = 256,
    ):
        self.img_root = img_root
        self.target_root = target_root
        self.crop_size = crop_size
        self.img_list = os.listdir(self.img_root)
        self.img_list.sort()


    def __getitem__(self, idx):

        x_path = os.path.join(self.img_root, self.img_list[idx])
        y_path = os.path.join(self.target_root, f"segmim_{self.img_list[idx].split('_')[1]}")
        x = read_image(x_path)
        y = read_image(y_path)

        # transform
        seed = torch.random.seed()

        x = x.to(torch.float) / 255.0
        torch.random.manual_seed(seed)
        x = transforms.RandomCrop(self.crop_size)(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.244, 0.225])(x)

        y = y.to(torch.float) / 255.0
        torch.random.manual_seed(seed)
        y = transforms.RandomCrop(self.crop_size)(y)

        return x, y

    def __len__(self):
        return self.img_list.__len__()


if __name__ == '__main__':
    td = Flower102Seg(
        img_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\train',
        target_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\mask',
    )

    x, y = td[0]
    plt.figure()
    plt.imshow(x.permute(1, 2, 0))
    plt.figure()
    plt.imshow(y.permute(1, 2, 0))
    plt.show()
    print(x.shape)
    print(y.shape)
    print(x)
    print(torch.max(y))

