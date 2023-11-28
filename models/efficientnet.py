# models/efficientnet.py

import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes=55):
        super(EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
