import cv2
import os
import re

def natural_sort_key(s):
    # Define a key for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def images_to_video(image_dir, output_video, duration):
    # Get list of image files in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg", ".tif"))]
    
    # Sort images by filename
    images.sort(key=natural_sort_key)
    
    print(images)

    # Calculate the FPS
    num_images = len(images)
    fps = num_images / duration

    # Read the first image to get the dimensions
    first_image = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_dir, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        resized_img = cv2.resize(img, (width, height))
        video.write(resized_img)
    
    # Release the VideoWriter object
    video.release()

    print(f"Video saved as {output_video}")

# Example usage
image_dir = './new_3d_predict_v2'  # Directory containing images
output_video = './new_3d_cell_v2.mp4'  # Output video file path
duration = 9  # Duration in seconds

images_to_video(image_dir, output_video, duration)