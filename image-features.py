import torch
import torch.nn as nn
from torchvision import models

import config

class ImageFeaturesNet(nn.Module):
    def __init__(self):
        super(ImageFeaturesNet, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)
        # save output from layer 4 of resnet => (14x14x2048)

    def forward(self, x):
        self.model(x)
        return self.buffer

def get_transform(target_size, scale_fraction=1.0):
    return transforms.Compose([
        transforms.Resize(int(target_size / scale_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def create_coco_loader(path):
    transform = get_transform(config.image_size, config.scale_fraction)
    data_loader = torch.utils.data.DataLoader(
        data.CocoImages(path, transform=transform),
        batch_size=config.image_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ImageFeaturesNet()
    net.eval()
    net.to(device)

    train_loader = create_coco_loader(config.train_path)
    val_loader = create_coco_loader(config.val_path)