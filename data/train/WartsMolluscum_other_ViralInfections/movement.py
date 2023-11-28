import os
import shutil

# Define your root directory here
#root_dir = '/home/saifm/dataset/train/Psoriasis_pictures_Lichen_Planus_related_diseases'
root_dir = os.getcwd()

# Get all jpg files in the root directory
jpg_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

for jpg_file in jpg_files:
    # Split the file name on hyphens
    parts = jpg_file.split('-')

    # Rejoin the parts to form the directory name, including everything up to the second hyphen
    # This will create directory names like 'atypical-nevi', 'atypical-nevi-dermoscopy', 'becker-nevus', etc.
    directory_name = '-'.join(parts[:2]) if len(parts) > 2 else parts[0]

    # Construct the directory path
    dir_path = os.path.join(root_dir, directory_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")

    # Copy the file to the directory
    shutil.copy(os.path.join(root_dir, jpg_file), os.path.join(dir_path, jpg_file))
    print(f"Copied {jpg_file} to {dir_path}/")

