# evaluate.py
import torch
from sklearn.metrics import classification_report
from utils.dataloader import load_data
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet
from models.efficientnet import EfficientNet
from models.vit import ViT
from utils.metrics import calculate_metrics

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

def evaluate_model(model, test_loader, model_name, device, save_to_file=True):
    model.eval()
    model.to(device)  # Move model to the device (GPU or CPU)
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Calculate metrics using the utility function
    accuracy, f1, precision, recall, roc_auc = calculate_metrics(all_labels, all_probs, all_preds)

    # Save metrics to a log file if required
    if save_to_file:
        log_file = f'evaluation_{model_name}.log'
        with open(log_file, 'w') as file:
            file.write(f"Model: {model_name}\n")
            file.write("Classification Report:\n")
            file.write(classification_report(all_labels, all_preds))
            file.write(f"ROC-AUC Score: {roc_auc}\n")
        print(f"Metrics saved to {log_file}")

    # Return the metrics for further use in comparison
    return accuracy, f1, precision, recall, roc_auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability

    model_name = "AlexNet"  # Replace with the model you want to evaluate
    num_classes = 55  # Number of classes in your dataset

    test_loader = load_data('../data/test', batch_size=32, is_train=False)
    model = select_model(model_name, num_classes)
    model.load_state_dict(torch.load(f"{model_name}.pth", map_location=device))
    model = model.to(device)  # Move model to the device
    model.eval()

    # Evaluate the model
    evaluate_model(model, test_loader, model_name, device)

if __name__ == "__main__":
    main()

