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


def get_revise_deeplabv3_mobilenet(pretrained: bool = True):
    """
    get revise deeplabv3_mobilenet_v3_large model with pretrained weights
    :return:
    """
    # model definition
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights if pretrained else None)

    # model revising
    model.classifier[-1] = torch.nn.Conv2d(256, 1, 1)
    if pretrained:
        model.aux_classifier[-1] = torch.nn.Conv2d(10, 1, 1)
    # model.classifier[-3] = torch.nn.BatchNorm2d(256)
    # print(model)
    return model


if __name__ == '__main__':

    saving_path = r'E:\Education\Master_Course\Image_process\group_project\training_record'

    epochs = 10
    lr = 1e-4

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    print(f'\nUsing {device} device.\n')

    # model
    model = get_revise_deeplabv3_mobilenet(pretrained=True)

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # to device
    model.to(device)
    criterion.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # dataset
    train_dataset = FlowerSegDataset.Flower102Seg(
        img_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\train',
        target_root=r'E:\Education\Master_Course\Image_process\group_project\dataset\flower102\mask',
        crop_size=400,
    )
    training_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 记录 loss，分割结果
    loss_epochs_list = utils.LossSaver()

    # training
    total_bar = tqdm(range(0, epochs), desc='[Total Progress]')
    for epoch in total_bar:
        model.train()
        loss_epoch = utils.AverageMeter()
        per_epoch_bar = tqdm(training_dataloader, desc=f'[Epoch {epoch}]', leave=False)
        for x, y in per_epoch_bar:
            x, y = x.to(device), y.to(device)
            output = model(x)['out']
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.update(loss.item())
            per_epoch_bar.set_postfix(loss=loss.item())

        loss_epochs_list.append(loss_epoch.avg)

        # save result
        save_image([output[0]], os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_output.png'))
        save_image([y[0]], os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_target.png'))
        save_image([x[0]], os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_img.png'))

        # plt.imsave(
        #     os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_img.png'),
        #     x[0].permute(1, 2, 0).cpu().numpy(),
        # )
        # plt.imsave(
        #     os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_target.png'),
        #     y[0][0].cpu().numpy(),
        # )
        # plt.imsave(
        #     os.path.join(saving_path, 'interval_result', f'epoch_{epoch}_output.png'),
        #     output[0][0].detach().cpu().numpy(),
        # ) # 这段暂时有问题

        # plt.figure()
        # plt.imshow(output[0][0].detach().cpu())
        # plt.figure()
        # plt.imshow(y[0][0].cpu())
        # plt.figure()
        # plt.imshow(x[0].permute(1, 2, 0).cpu())
        # plt.show()

        # save model
        if epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saving_path, f'checkpoint_epoch_{epoch}.pth'),
            )

    # save loss
    loss_epochs_list.save_to_file(
        os.path.join(saving_path, f'loss_epochs_{epochs}_lr_{lr}.npy')
    )

