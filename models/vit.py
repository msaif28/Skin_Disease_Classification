import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViT(nn.Module):
    def __init__(self, num_classes=55, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViT, self).__init__()

        # Load the pre-trained Vision Transformer configuration
        self.config = ViTConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)

        # Load the pre-trained Vision Transformer model
        self.vit = ViTModel(self.config)

        # Replace the classifier head
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(x).last_hidden_state
        cls_output = outputs[:, 0, :]  # Taking the output corresponding to the [CLS] token
        return self.classifier(cls_output)

    def load_pretrained_weights(self, path):
        # Load pre-trained weights into the model
        self.load_state_dict(torch.load(path), strict=False)

# Optional testing block
if __name__ == "__main__":
    model = ViT(num_classes=55)  # Example with 55 classes
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    print(model(dummy_input).shape)  # Output shape

