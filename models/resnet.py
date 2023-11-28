# models/resnet.py

import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=55):  # Set num_classes to 55 for your dataset
        super(ResNet, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)  # You can also use resnet34, resnet50, etc.
        # Replace the fully connected layer (fc) for your number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
