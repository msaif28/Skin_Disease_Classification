import os

# Define the root directory
root_dir = '/home/saifm/dataset/dermonet_Saif/train'

total_parent_directories = 0
total_subdirectories = 0

print(f"{'Parent_ Classes Name':<60} | {'Total_Number_Of_Sub_Classes':<25}")

for entry in os.listdir(root_dir):
    path = os.path.join(root_dir, entry)
    if os.path.isdir(path):
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"{entry:<60} | {len(subdirs):<25}")
        total_parent_directories += 1
        total_subdirectories += len(subdirs)

# Print total counts
print("\nTotal_Parent_ Classes:", total_parent_directories)
print("Total_Sub_Classes_Parent:", total_subdirectories)
