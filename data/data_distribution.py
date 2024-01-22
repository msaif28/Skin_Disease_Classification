import os
import shutil
import random

# Define base directory and original training images directory
base_dir = "/home/saifm/Skin_Disease_Classification-Jan2024/data"
source_dir = os.path.join(base_dir, "train_orig")  # original images

# Define where to save the train, test, and validation images
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

# Define image file extensions
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Ensure the train, test, and validation directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through each directory in the source directory
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)

    # Proceed only if it's a directory
    if os.path.isdir(subdir_path):
        # List all image files in the directory
        images = [f for f in os.listdir(subdir_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        # Check if there are any images to distribute
        if images:
            # Shuffle the images for random distribution
            random.shuffle(images)

            # Calculate the number of images for test, val and the remaining for train sets
            total_images = len(images)
            test_count = int(0.15 * total_images)
            val_count = int(0.15 * total_images)
            train_count = total_images - test_count - val_count  # the rest goes to train

            # Split images into train, test, and validation sets
            test_images = images[:test_count]
            val_images = images[test_count:test_count + val_count]
            train_images = images[test_count + val_count:]

            # Define function to copy images to target directory (changed from move to copy)
            def copy_images(image_list, source_subdir, target_subdir):
                os.makedirs(target_subdir, exist_ok=True)  # Ensure target directory exists
                for image in image_list:
                    source_path = os.path.join(source_subdir, image)
                    target_path = os.path.join(target_subdir, image)
                    shutil.copy(source_path, target_path)

            # Copy the images to respective directories
            copy_images(train_images, subdir_path, os.path.join(train_dir, subdir))
            copy_images(test_images, subdir_path, os.path.join(test_dir, subdir))
            copy_images(val_images, subdir_path, os.path.join(val_dir, subdir))

print("Images have been distributed into train, test, and val directories.")
