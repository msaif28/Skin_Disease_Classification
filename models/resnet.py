# models/resnet.py

import torch.nn as nn
import torch.hub

class ResNet(nn.Module):
    def __init__(self, num_classes=55):
        super(ResNet, self).__init__()
        # Load a pre-trained ResNet-18 model using torch.hub
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Replace the fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
