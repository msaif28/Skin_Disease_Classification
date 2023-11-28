# train.py
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils.dataloader import load_data
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet
from models.efficientnet import EfficientNet
from models.vit import ViT

def select_model(model_name, num_classes):
    if model_name == 'AlexNet':
        return AlexNet(num_classes=num_classes)
    elif model_name == 'VGG':
        return VGG(num_classes=num_classes)
    elif model_name == 'ResNet':
        return ResNet(num_classes=num_classes)
    elif model_name == 'EfficientNet':
        return EfficientNet(num_classes=num_classes)
    elif model_name == 'ViT':
        return ViT(num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")

def train_model(model, train_loader, criterion, optimizer, num_epochs, model_name, device):
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

    # Save the trained model
    try:
        torch.save(model.state_dict(), f"{model_name}.pth")
        print(f"Model saved as {model_name}.pth")
    except IOError as e:
        print(f"Error saving the model: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    model_name = "AlexNet"  # Replace with the desired model's name
    num_classes = 55  # Number of classes in your dataset

    train_loader = load_data('../data/train', batch_size=32)
    model = select_model(model_name, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # Set the number of epochs you want to train for

    train_model(model, train_loader, criterion, optimizer, num_epochs, model_name, device)

if __name__ == "__main__":
    main()