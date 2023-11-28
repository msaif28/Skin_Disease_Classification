import os

# Define the root directory
root_dir = '/home/saifm/dataset/dermonet_Saif/train'

total_parent_directories = 0
total_subdirectories = 0
total_images = 0

print(f"{'Parent Classes Name':<60} | {'Total Number Of Sub-Classes':<30} | {'Number of Images':<20}")

for entry in os.listdir(root_dir):
    path = os.path.join(root_dir, entry)
    if os.path.isdir(path):
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        subdir_images_count = 0
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f)) and f.lower().endswith('.jpg')]
            subdir_images_count += len(images)
        print(f"{entry:<60} | {len(subdirs):<30} | {subdir_images_count:<20}")
        total_parent_directories += 1
        total_subdirectories += len(subdirs)
        total_images += subdir_images_count

# Print total counts
print("\nTotal Parent Classes:", total_parent_directories)
print("Total Sub-Classes per Parent:", total_subdirectories)
print("Total Images:", total_images)
