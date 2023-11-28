import os

# Define the root directory
root_dir = '/home/saifm/dataset/dermonet_Saif/train'

# Initialize a list to store sub-class information (sub-class name, parent class, image count)
sub_class_info = []

# Iterate over each parent directory and count images for each sub-class
for parent_entry in os.listdir(root_dir):
    parent_path = os.path.join(root_dir, parent_entry)
    if os.path.isdir(parent_path):
        for sub_entry in os.listdir(parent_path):
            sub_path = os.path.join(parent_path, sub_entry)
            if os.path.isdir(sub_path):
                images = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f)) and f.lower().endswith('.jpg')]
                sub_class_info.append((parent_entry, sub_entry, len(images)))

# Sort the sub-classes by the number of images (in descending order) and take the top 50
top_50_sub_classes = sorted(sub_class_info, key=lambda x: x[2], reverse=True)[:50]

# Define column widths for the table
parent_col_width = 50
sub_class_col_width = 50
image_col_width = 20

# Print the header
header_format = f"{{:<{parent_col_width}}} | {{:<{sub_class_col_width}}} | {{:>{image_col_width}}}"
print(header_format.format("Parent Class", "Sub-Class", "Number of Images"))

# Print the top 50 sub-classes with parent class and image counts
for parent, sub_class, count in top_50_sub_classes:
    print(header_format.format(parent, sub_class, count))
