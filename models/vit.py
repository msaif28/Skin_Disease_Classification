# models/vit.py

import torch.nn as nn
import torchvision.models as models

class ViT(nn.Module):
    def __init__(self, num_classes=55):  # Adjust num_classes based on your dataset
        super(ViT, self).__init__()
        # Load a pre-trained Vision Transformer model, ViT-B/16 as an example
        self.vit = models.vit_b_16(pretrained=True)
        # Replace the head with a new one, adjusting to your number of classes
        self.vit.heads[0] = nn.Linear(self.vit.heads[0].in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
