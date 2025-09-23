import os
import cv2
import matplotlib.pyplot as plt
import random
import yaml
import numpy as np

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file by finding it relative to the script's location."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def get_image_paths(root_dir):
    """
    Lists and verifies all image files in the specified directory using OpenCV.
    """
    image_paths = []
    # Get the absolute path for the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    data_root_full_path = os.path.join(project_root, root_dir)
    
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jp2')
    
    for dirpath, _, filenames in os.walk(data_root_full_path):
        for filename in filenames:
            if filename.lower().endswith(supported_formats):
                full_path = os.path.join(dirpath, filename)
                try:
                    # Use OpenCV's imread, which can now handle .jp2 with imagecodecs installed
                    img = cv2.imread(full_path)
                    if img is not None:
                        image_paths.append(full_path)
                    else:
                        print(f"Warning: Corrupt or unreadable file: {full_path}")
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return image_paths

def visualize_sample_images(image_paths, num_samples=5):
    """Displays a random sample of images from the dataset."""
    if not image_paths:
        print("No images to visualize.")
        return

    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    plt.figure(figsize=(15, 8))
    for i, path in enumerate(sample_paths):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_rgb)
        plt.title(os.path.basename(os.path.dirname(path)), fontsize=8)
        plt.axis('off')
        
    plt.suptitle("Sample Images from Dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        config = load_config()
        data_root = config['paths']['data_root']
        
        print("Starting data exploration...")
        all_images = get_image_paths(data_root)
        print(f"Found {len(all_images)} verified images.")
        
        print("\nVisualizing a random sample...")
        visualize_sample_images(all_images)
        print("Data exploration complete.")
    except Exception as e:
        print(f"An error occurred: {e}")