import json

import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt

from torchinfo import summary

if __name__ == '__main__':

    img = read_image(r"D:\AppData\Tencent\763851927\FileRecv\MobileFile\IMG_1565(20231222-123439).JPG")

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    print(preprocess)

    batch = preprocess(img).unsqueeze(0)
    # x = img.to(torch.float) / 255.0
    # # x = torchvision.transforms.RandomCrop(256)(x)
    # batch = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.244, 0.225])(x).unsqueeze(0)

    print(batch.shape)

    prediction = model(batch)["out"]  # 输出会插值回原始大小
    normalized_masks = prediction.softmax(dim=1)

    print(normalized_masks.shape)

    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    argmax_masks = normalized_masks[0].argmax(dim=0)

    print(argmax_masks.shape)

    print(class_to_idx)
    # mask = normalized_masks[0, class_to_idx["pottedplant"]]
    # print(mask.shape)
    # to_pil_image(argmax_masks.int()).show()

    plt.figure()
    plt.imshow(argmax_masks.int())
    plt.show()

    summary(model, input_size=(1, 3, 520, 693))

    # # model revising
    # print(model.classifier[-1])
    # model.classifier[-1] = torch.nn.Conv2d(256, 2, 1)
    # 继续修改 aux 中的最后一层
    # print(model.classifier)


