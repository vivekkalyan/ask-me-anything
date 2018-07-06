import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import h5py

import data
import config


class ImageFeaturesNet(nn.Module):
    def __init__(self):
        super(ImageFeaturesNet, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.avgpool = EmptyLayer()
        self.model.fc = EmptyLayer()

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)
        # save output from layer 4 of resnet => (14x14x2048)

    def forward(self, x):
        self.model(x)
        return self.buffer


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


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


def create_preprocessed_file(model, input_images_path, output_file_path):
    loader = create_coco_loader(input_images_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(output_file_path, libver='latest') as f:
        features = f.create_dataset(
            'features', shape=features_shape, dtype='float16')
        coco_ids = f.create_dataset('ids', shape=(
            len(loader.dataset),), dtype='int32')

        a = 0
        b = 0
        for _, (ids, images) in enumerate(loader):
            out = model(images)

            b = a + images.size(0)
            features[a:b, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[a:b] = ids.numpy().astype('int32')
            a = b


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ImageFeaturesNet()
    net.eval()
    net.to(device)

    create_preprocessed_file(
        net, 'vqa/mscoco/small_sample', config.img_feature_path)


if __name__ == '__main__':
    main()
