import torch
import numpy as np
import os
import sys
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils.dataloader import load_data

# Import your model modules
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet
from models.efficientnet import EfficientNet
from models.vit import ViT

# Update this to the path where your models are saved
model_save_path = '/home/saifm/Experiments/Experiment01/output'
# Update this to the path of your test dataset
test_data_path = '/home/saifm/Experiments/Experiment01/data/test'

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
        return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")

def load_model(model_name, num_classes, device):
    model = select_model(model_name, num_classes)
    model_path = os.path.join(model_save_path, f"{model_name}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_and_report(model, test_loader, device, num_classes, class_idx_to_name, model_name):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)

    # Binarize the labels for ROC AUC and ensure all_probs is a numpy array
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)  # Convert list of probabilities to a numpy array

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_idx_to_name[idx] for idx in range(num_classes)],
                yticklabels=[class_idx_to_name[idx] for idx in range(num_classes)])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # Plot all ROC curves
    plt.figure(figsize=(20, 20))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'gray', 'olive', 'cyan'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(class_idx_to_name[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for each class')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curves_{model_name}.png')
    plt.close()

    # Save ROC AUC scores to a text file
    roc_auc_file_path = f'roc_auc_report_{model_name}.txt'
    with open(roc_auc_file_path, 'w') as file:
        for class_idx, roc_score in roc_auc.items():
            roc_auc_line = f"ROC AUC for {class_idx_to_name[class_idx]}: {roc_score}\n"
            print(roc_auc_line, end='')
            file.write(roc_auc_line)

    # Print and save the classification report
    classification_rep = classification_report(all_labels, all_preds, target_names=[class_idx_to_name[idx] for idx in range(num_classes)])
    print(f"Classification Report for {model_name}:\n{classification_rep}")
    report_file_path = f'classification_report_{model_name}.txt'
    with open(report_file_path, 'w') as file:
        file.write(classification_rep)

    # Return labels, predictions, and probabilities
    return all_labels, all_preds, all_probs

def main():
    # Set the device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtain class names by reading the directory structure
    class_names = sorted(os.listdir(test_data_path))
    class_idx_to_name = {idx: name for idx, name in enumerate(class_names)}
    num_classes = len(class_idx_to_name)

    # Load your test data
    test_loader = load_data(test_data_path, batch_size=64, is_train=False)

    # List of models to evaluate
    models_to_evaluate = ['AlexNet', 'VGG', 'ResNet', 'EfficientNet', 'ViT']

    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"Evaluating {model_name}")
        # Load the model
        model = load_model(model_name, num_classes, device)
        # Perform evaluation
        evaluate_and_report(model, test_loader, device, num_classes, class_idx_to_name, model_name)

# Entry point of the script
if __name__ == "__main__":
    main()
