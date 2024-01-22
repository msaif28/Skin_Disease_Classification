import torch
from torchvision import transforms
from PIL import Image
from lime import lime_image
import matplotlib.pyplot as plt
from models.alexnet import AlexNet  # Replace with your model
from utils.dataloader import load_data  # Replace with your data loader method
import numpy as np
from skimage.segmentation import mark_boundaries

# Load your trained model
model = AlexNet(num_classes=55)  # Replace with your model and number of classes
model.load_state_dict(torch.load('path/to/saved/model.pth', map_location=torch.device('cpu')))
model.eval()

# Define a function to preprocess images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# LIME explanation for a single image
def explain_image(model, image_tensor, original_image, top_labels):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(original_image), 
                                             lambda x: model(torch.tensor(x).permute(0, 3, 1, 2)).detach().numpy(), 
                                             top_labels=top_labels, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, hide_rest=False)

    # Display original image and explanation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title('LIME Explanation')
    plt.axis('off')

    plt.show()

# Path to your test dataset
test_data_path = 'path/to/test/dataset'
test_dataset = load_data(test_data_path, batch_size=1, is_train=False)

# Example usage for a subset of test images
for images, _ in test_dataset:  # Assuming test_dataset is a DataLoader
    original_image = transforms.ToPILImage()(images[0])
    preprocessed_image = preprocess_image(original_image)
    explain_image(model, preprocessed_image, original_image, top_labels=5)
    break  # Remove this line to explain more images
