# models/alexnet.py

import torch.nn as nn
import torch.hub

class AlexNet(nn.Module):
    def __init__(self, num_classes=55):  # Adjust num_classes based on your dataset
        super(AlexNet, self).__init__()
        # Load a pre-trained AlexNet model using torch.hub
        self.alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)