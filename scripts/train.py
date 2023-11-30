import sys
import os

# Ensure the script can find custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import DataParallel
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
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")

    # CSV file for metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    with open(metrics_path, 'w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy'])

    best_accuracy = 0.0
    model.to(device)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=total_loss / (i + 1), accuracy=100. * correct / total)

        average_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}. Accuracy: {accuracy:.2f}%.")

        # Write metrics to CSV after each epoch
        with open(metrics_path, 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([epoch + 1, average_loss, accuracy])

        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_save_path = os.path.join(output_dir, f"{model_name}_best.pth")
            torch.save(model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(), best_model_save_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%.")

    # Save the final model
    final_model_save_path = os.path.join(output_dir, f"{model_name}_{num_epochs}_epochs.pth")
    torch.save(model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(), final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")

def main():
    if torch.cuda.is_available():
        print("Available GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("GPU", i, ":", torch.cuda.get_device_name(i))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "AlexNet"  # Replace with your chosen model
    num_classes = 55  # Replace with the number of classes in your dataset
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train'))
    train_loader = load_data(data_dir, batch_size=32)
    model = select_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000  # Set your desired number of epochs

    train_model(model, train_loader, criterion, optimizer, num_epochs, model_name, device)

if __name__ == "__main__":
    main()
