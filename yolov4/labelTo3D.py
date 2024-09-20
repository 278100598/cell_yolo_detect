import os
import cv2
import numpy as np
from PIL import Image
import re

# Define the colors
colors = {
    0: (0, 255, 255),  # yellow
    1: (0, 0, 255),    # red
    2: (255, 0, 0)     # blue
}

def natural_sort_key(s):
    # Define a key for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Function to draw bounding boxes on images
def draw_bounding_boxes(image, labels, colors):
    h, w = image.shape[:2]
    for label in labels:
        cls, x_center, y_center, width, height = map(float, label.strip().split())
        cls = int(cls)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        color = colors.get(cls, (255, 255, 255))  # default to white if class not specified
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

# Directories
label_dir = './new_3d_predict_label_v2'  # Directory containing label text files
image_dir = '../datasets/new_3d'  # Directory containing .tif image files
output_dir = './new_3d_predict_v2'  # Directory to save output .jpg files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
label_files.sort(key=natural_sort_key)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
print("label file:", label_files)
#print(len(label_files))
print("image file:", image_files)
#print(len(image_files))
if len(label_files) != len(image_files):
    print("The number of label files does not match the number of image files.")
    exit(1)
# Process each label file
for label_file, image_file in zip(label_files, image_files):
    base_name = os.path.splitext(label_file)[0]
    image_path = os.path.join(image_dir, image_file)
    output_file = os.path.join(output_dir, base_name + '.jpg')

    # Read the image
    image = cv2.imread(image_path)

    # Read the label file
    with open(os.path.join(label_dir, label_file), 'r') as f:
        labels = f.readlines()

    # Draw bounding boxes on the image
    image = draw_bounding_boxes(image, labels, colors)

    # Save the image as a .jpg file
    cv2.imwrite(output_file, image)
    print(f"Output saved to {output_file}")