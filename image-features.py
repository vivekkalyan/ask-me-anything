import torch
from torchvision import models

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