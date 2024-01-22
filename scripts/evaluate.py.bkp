import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from sklearn.metrics import classification_report
import warnings
from utils.dataloader import load_data
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet
from models.efficientnet import EfficientNet
from models.vit import ViT
import timm
from utils.metrics import calculate_metrics

# Suppress specific known warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        return timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")

def evaluate_model(model, test_loader, model_name, device, output_dir, save_to_file=True):
    model.eval()
    model.to(device)
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)

    accuracy, f1, precision, recall, roc_auc = calculate_metrics(all_labels, all_preds, all_probs)

    if save_to_file:
        log_file = os.path.join(output_dir, f'evaluation_{model_name}.log')
        with open(log_file, 'w') as file:
            file.write(f"Model: {model_name}\n")
            file.write("Classification Report:\n")
            file.write(classification_report(all_labels, all_preds))
            file.write(f"ROC-AUC Score: {roc_auc}\n")
        print(f"Metrics saved to {log_file}")

    return accuracy, f1, precision, recall, roc_auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 55
    test_loader = load_data('/home/saifm/Skin_Disease_Classification-Jan2024/data/test', batch_size=64, is_train=False)
    models_to_evaluate = ['AlexNet', 'VGG', 'ResNet', 'EfficientNet', 'ViT']
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the model files are named as '<model_name>_best_<timestamp>.pth'
    for model_name in models_to_evaluate:
        model = select_model(model_name, num_classes)

        # Find the latest model file by timestamp
        model_files = [f for f in os.listdir(output_dir) if f.startswith(model_name) and f.endswith('.pth')]
        model_files.sort(reverse=True)  # Latest file first
        if model_files:
            model_weight_filename = model_files[0]  # Pick the latest file
            model_path = os.path.join(output_dir, model_weight_filename)

            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                accuracy, f1, precision, recall, roc_auc = evaluate_model(model, test_loader, model_name, device, output_dir)
                with open(os.path.join(output_dir, 'model_evaluation_metrics.csv'), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([model_name, accuracy, f1, precision, recall, roc_auc])
            except Exception as e:
                print(f"Error loading model {model_name} from {model_weight_filename}: {e}")
        else:
            print(f"No model file found for {model_name}")

if __name__ == "__main__":
    main()
