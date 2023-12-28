import json

import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchinfo import summary

from finetune import get_revise_deeplabv3_mobilenet

if __name__ == '__main__':

    # img = read_image(r"E:\Education\Master_Course\Image_process\group_project\dataset\flower102\jpg\image_02890.jpg")
    img = read_image(r".\DSC00238.JPG")

    model = get_revise_deeplabv3_mobilenet(pretrained=False)

    checkpoint = torch.load(
        r'E:\Education\Master_Course\Image_process\group_project\training_record\checkpoint_epoch_9.pth'
    )

    model.load_state_dict(checkpoint)

    img = img.to(torch.float) / 255.0
    # img = transforms.RandomCrop(400)(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.244, 0.225])(img)

    with torch.no_grad():

        model.eval()
        batch = img.unsqueeze(0)

        prediction = torch.nn.functional.sigmoid(model(batch)["out"])  # 输出会插值回原始大小
        mask = torch.where(prediction > 0.5, torch.ones_like(prediction), torch.zeros_like(prediction))

        plt.figure()
        plt.imshow(prediction[0][0])
        plt.figure()
        plt.imshow(mask[0][0])
        plt.figure()
        plt.imshow(img.permute(1, 2, 0))
        plt.show()



