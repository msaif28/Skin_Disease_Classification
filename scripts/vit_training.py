import sys
import os
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from utils.dataloader import load_data
import timm
from datetime import datetime

def select_model(model_name, num_classes):
    if model_name == 'ViT':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        for param in model.head.parameters():
            param.requires_grad = True
        return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name, device):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.csv")
    best_model_path = os.path.join(output_dir, f'{model_name}_best_{timestamp}.pth')
    
    with open(metrics_path, 'w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy'])

    best_val_accuracy = 0

    for epoch in range(num_epochs):
        if epoch == 300:  
            print("Reached 300 epochs, stopping training.")
            break

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

        scheduler.step()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                batch_loss = criterion(val_outputs, val_labels)
                val_loss += batch_loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = 100 * val_correct / val_total

        with open(metrics_path, 'a', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([epoch + 1, total_loss / total, correct / total, val_loss / val_total, val_accuracy])

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}, Val Loss: {val_loss/val_total:.4f}, Val Accuracy: {val_accuracy:.4f}')

    print('Finished Training')

def main():
    parser = argparse.ArgumentParser(description='Train a specific model on a specific GPU')
    parser.add_argument('--model_name', type=str, required=True, choices=['ViT', 'OtherModels'], help='Model to train')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = select_model(args.model_name, num_classes=55)
    model.to(device)

    train_loader = load_data('../data/train', batch_size=64, is_train=True)
    val_loader = load_data('../data/val', batch_size=64, is_train=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 1000, args.model_name, device)

if __name__ == "__main__":
    main()

