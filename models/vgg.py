# models/vgg.py

import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, num_classes=55):  # Adjust num_classes based on your dataset
        super(VGG, self).__init__()
        # Load a pre-trained VGG model, VGG16 as an example
        self.vgg = models.vgg16(pretrained=True)
        # Replace the classifier with a new one, adjusting the final layer to your number of classes
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
