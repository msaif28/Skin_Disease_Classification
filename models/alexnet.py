# models/alexnet.py

import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # Adjust num_classes based on your dataset
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)