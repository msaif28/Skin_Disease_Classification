import os
import torch

def save_weights_to_txt(pth_file_path, txt_file_path):
    state_dict = torch.load(pth_file_path, map_location='cpu')
    with open(txt_file_path, 'w') as f:
        for name, weights in state_dict.items():
            # Write the layer name
            f.write(f"Layer: {name}\n")
            # Write the weights as a flattened list to save space
            weights_flat = weights.cpu().numpy().flatten()
            weights_str = ', '.join([str(w) for w in weights_flat])
            f.write(weights_str + "\n\n")

def convert_pth_to_txt(directory):
    # Scan the directory for .pth files
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            # Construct full file paths
            pth_file_path = os.path.join(directory, filename)
            txt_file_path = os.path.join(directory, filename.replace('.pth', '.txt'))

            # Save the weights and biases to a .txt file
            save_weights_to_txt(pth_file_path, txt_file_path)
            print(f"Saved weights and biases from {pth_file_path} to {txt_file_path}")

# Get the current working directory
current_directory = os.getcwd()
convert_pth_to_txt(current_directory)