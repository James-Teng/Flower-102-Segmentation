import json
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchinfo import summary

import FlowerSegDataset
import utils
import finetune


def IoU(pred, mask):
    intersection = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask) - intersection
    return intersection / union if union != 0 else 0


if __name__ == '__main__':

    checkpoint_path = r'E:\Education\Master_Course\Image_process\group_project\training_record\checkpoint_epoch_9.pth'

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    print(f'\nUsing {device} device.\n')

    # model
    model = finetune.get_revise_deeplabv3_mobilenet(pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    # to device
    model.to(device)

    # dataset
    test_dataset = FlowerSegDataset.Flower102Seg(
        img_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\test',
        target_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\mask',
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    average_IoU = utils.AverageMeter()

    with torch.no_grad():
        model.eval()
        eval_bar = tqdm(test_dataloader, desc=f'[Eval]', leave=False)
        for x, y in eval_bar:
            x, y = x.to(device), y.to(device)
            output = torch.sigmoid(model(x)['out'])
            mask = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
            iou = IoU(mask, y)
            assert iou <= 1, f'{torch.sum(y * mask)} {torch.sum(y)} {torch.sum(mask)}'
            average_IoU.update(iou)
            eval_bar.set_postfix(IoU=iou)

    print(f'count: {average_IoU.count}')
    print(f'sum: {average_IoU.sum}')
    print(f'average IoU: {average_IoU.avg}')
    # pretrain ep_0 average IoU: 0.8010552525520325
    # from scratch ep_0 average IoU: 0.8139369487762451
    # from scratch ep_9 average IoU: 0.8080199956893921





