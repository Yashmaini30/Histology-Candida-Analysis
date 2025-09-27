import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import random
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def load_annotations(annotations_path):
    """Load bounding box annotations from CSV file."""
    return pd.read_csv(annotations_path)

def find_image_path(image_name, data_root):
    """Find the full path of an image in the data directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    data_root_full = os.path.join(project_root, data_root)
    
    for root, dirs, files in os.walk(data_root_full):
        for file in files:
            if file == image_name:
                return os.path.join(root, file)
    return None

def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image."""
    image_copy = image.copy()
    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        # Add label
        cv2.putText(image_copy, row['label'], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_copy

def visualize_sample_annotations(annotations_df, data_root, num_samples=6):
    """Visualize random sample of images with their annotations."""
    # Get unique image names
    unique_images = annotations_df['image_path'].unique()
    sample_images = random.sample(list(unique_images), min(num_samples, len(unique_images)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, image_name in enumerate(sample_images):
        # Find full image path
        image_path = find_image_path(image_name, data_root)
        
        if image_path is None:
            print(f"Warning: Could not find image {image_name}")
            continue
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        # Get annotations for this image
        image_annotations = annotations_df[annotations_df['image_path'] == image_name]
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image, image_annotations)
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[i].imshow(image_rgb)
        axes[i].set_title(f"{image_name}\n{len(image_annotations)} Candida regions")
        axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle("Sample Images with Candida Annotations", fontsize=16)
    plt.tight_layout()
    plt.show()

def get_annotation_statistics(annotations_df):
    """Print statistics about the annotations."""
    print("=== ANNOTATION STATISTICS ===")
    print(f"Total annotations: {len(annotations_df)}")
    print(f"Unique images: {annotations_df['image_path'].nunique()}")
    print(f"Average annotations per image: {len(annotations_df) / annotations_df['image_path'].nunique():.2f}")
    
    # Bounding box size statistics
    annotations_df['width'] = annotations_df['xmax'] - annotations_df['xmin']
    annotations_df['height'] = annotations_df['ymax'] - annotations_df['ymin']
    annotations_df['area'] = annotations_df['width'] * annotations_df['height']
    
    print(f"\nBounding Box Size Statistics:")
    print(f"Average width: {annotations_df['width'].mean():.2f} px")
    print(f"Average height: {annotations_df['height'].mean():.2f} px")
    print(f"Average area: {annotations_df['area'].mean():.2f} px²")
    print(f"Min area: {annotations_df['area'].min():.2f} px²")
    print(f"Max area: {annotations_df['area'].max():.2f} px²")
    
    # Label distribution
    print(f"\nLabel Distribution:")
    print(annotations_df['label'].value_counts())
    
    return annotations_df

def visualize_single_image_detailed(image_name, annotations_df, data_root):
    """Show detailed view of a single image with all its annotations."""
    image_path = find_image_path(image_name, data_root)
    
    if image_path is None:
        print(f"Could not find image: {image_name}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Get annotations for this image
    image_annotations = annotations_df[annotations_df['image_path'] == image_name]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Original: {image_name}")
    ax1.axis('off')
    
    # Image with annotations
    image_with_boxes = draw_bounding_boxes(image, image_annotations)
    ax2.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"With {len(image_annotations)} Candida Detections")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Annotations for {image_name}:")
    for i, (_, row) in enumerate(image_annotations.iterrows()):
        print(f"  {i+1}: ({row['xmin']}, {row['ymin']}) to ({row['xmax']}, {row['ymax']}) - Area: {(row['xmax']-row['xmin'])*(row['ymax']-row['ymin'])} px²")

if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()
        
        # Load annotations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        annotations_path = os.path.join(project_root, 'results', 'automated_detection', 'bounding_boxes.csv')
        
        print("Loading annotations...")
        annotations_df = load_annotations(annotations_path)
        
        # Show statistics
        annotations_df = get_annotation_statistics(annotations_df)
        
        # Visualize samples
        print("\nGenerating sample visualizations...")
        visualize_sample_annotations(annotations_df, config['paths']['data_root'])
        
        # Show detailed view of first image
        first_image = annotations_df['image_path'].iloc[0]
        print(f"\nShowing detailed view of: {first_image}")
        visualize_single_image_detailed(first_image, annotations_df, config['paths']['data_root'])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()