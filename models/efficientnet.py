# models/efficientnet.py

import torch.nn as nn
import torch.hub

class EfficientNet(nn.Module):
    def __init__(self, num_classes=55):
        super(EfficientNet, self).__init__()
        # Load a pre-trained EfficientNet-B6 model using torch.hub
        self.efficientnet = torch.hub.load('pytorch/vision:v0.12.0', 'efficientnet_b6', pretrained=True)
        # Replace the classifier's final layer
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)