# models/vgg.py

import torch.nn as nn
import torch.hub

class VGG(nn.Module):
    def __init__(self, num_classes=55):
        super(VGG, self).__init__()
        # Load a pre-trained VGG-16 model using torch.hub
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        # Replace the classifier's final layer
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)