import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import csv
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet
from models.efficientnet import EfficientNet
from models.vit import ViT

# Ensure the script can find custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 55
    test_dir = '../data/test'  # Directory containing test images

    # List of models to predict with
    models_to_predict = ['AlexNet', 'VGG', 'ResNet', 'EfficientNet', 'ViT']

    for model_name in models_to_predict:
        print(f"Making predictions with {model_name}")

        model = select_model(model_name, num_classes)
        model_path = f"{model_name}.pth"  # Adjust the path if necessary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    prediction = predict_image(image_path, model, device)
                    print(f'Model: {model_name}, Image: {image_path}, Predicted Class: {prediction}')

if __name__ == "__main__":
    main()