import os
import cv2
import yaml
import numpy as np
from skimage import morphology
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import pandas as pd
from image_utils import load_image_robust, get_image_paths_robust

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def get_image_paths(root_dir):
    """Lists and verifies all image files in the specified directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    data_root_full_path = os.path.join(project_root, root_dir)
    
    return get_image_paths_robust(data_root_full_path)

def color_segmentation(image):
    """
    Performs color-based segmentation to find dark, reddish/purplish regions.
    Returns a binary mask.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Refined range for Candida based on your image
    # Note: Hue (H) is adjusted to a narrower band around the specific purple of the hyphae
    # Saturation (S) is increased to ignore lighter, less-stained areas
    # Value (V) is adjusted to capture the darker elements
    lower_purple = np.array([130, 100, 20])  # Tighter H and higher S, lower V to capture dark areas
    upper_purple = np.array([160, 255, 150]) # Tighter H and higher S, lower V to capture dark areas
    
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Post-processing to clean up the mask
    # Erosion to remove small noise and separate touching objects
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Dilation to restore the shape of the remaining objects
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def find_bounding_boxes(mask):
    """Finds bounding boxes for segmented regions and filters by size."""
    # This part remains mostly the same, but the filtering is crucial now.
    # The minimum contour area should be large enough to filter out noise,
    # but small enough to capture individual cells or small clusters.
    min_area = 50 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area] 
    return boxes

def find_filtered_bounding_boxes(mask):
    """
    Finds bounding boxes for segmented regions and filters them based on
    size and a simple shape approximation (solidity or circularity).
    """
    min_area = 50
    min_solidity = 0.5  # A measure of how "filled in" the shape is

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            # Calculate solidity as a shape filter
            # Solidity = Contour Area / Bounding Box Area
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > min_solidity:
                    x, y, w, h = cv2.boundingRect(c)
                    filtered_boxes.append((x, y, w, h))
    
    return filtered_boxes

def process_single_image(image_path, output_annotations_dir, output_masks_dir):
    """
    Process a single image: segment, find bounding boxes, and save results.
    """
    try:
        # Load image using robust loading
        img = load_image_robust(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return []
        
        # Perform segmentation
        mask = color_segmentation(img)
        
        # Find bounding boxes
        boxes = find_filtered_bounding_boxes(mask)
        
        # Prepare results
        image_name = os.path.basename(image_path)
        results = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            results.append({
                'image_path': image_name,
                'label': 'candida',
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h,
                'confidence': 1.0  # Since this is rule-based, we set high confidence
            })
        
        # Save mask if masks directory is provided
        if output_masks_dir:
            os.makedirs(output_masks_dir, exist_ok=True)
            mask_filename = os.path.splitext(image_name)[0] + '.png'
            mask_path = os.path.join(output_masks_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
        
        print(f"Processed {image_name}: Found {len(boxes)} Candida regions")
        return results
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def process_and_save(image_paths, output_annotations_dir, output_masks_dir=None):
    """
    Process multiple images and save all results.
    """
    os.makedirs(output_annotations_dir, exist_ok=True)
    if output_masks_dir:
        os.makedirs(output_masks_dir, exist_ok=True)
    
    all_results = []
    
    # Process each image
    for image_path in image_paths:
        results = process_single_image(image_path, output_annotations_dir, output_masks_dir)
        all_results.extend(results)
    
        # Save all annotations to CSV
        if all_results:
            annotations_df = pd.DataFrame(all_results)
            output_csv = os.path.join(output_annotations_dir, 'automated_detections.csv')
            annotations_df.to_csv(output_csv, index=False)
            print(f"Saved {len(all_results)} detections to {output_csv}")
        else:
            print("No detections found in any images.")
        
        return all_results

def visualize_detection_results(image_path, results, mask=None):
    """
    Visualize detection results on a single image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Draw bounding boxes
    img_with_boxes = img.copy()
    for result in results:
        if result['image_path'] == os.path.basename(image_path):
            x1, y1 = int(result['xmin']), int(result['ymin'])
            x2, y2 = int(result['xmax']), int(result['ymax'])
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{result['label']} ({result['confidence']:.2f})"
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Create visualization
    fig_width = 20 if mask is not None else 10
    plt.figure(figsize=(fig_width, 8))
    
    if mask is not None:
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Segmentation Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Candida Regions")
        plt.axis('off')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Candida Regions")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_and_save(image_paths, annotations_dir, masks_dir):
    """Processes images, generates annotations, and saves them."""
    if not os.path.exists(annotations_dir): os.makedirs(annotations_dir)
    if not os.path.exists(masks_dir): os.makedirs(masks_dir)

    all_boxes_data = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping corrupt file: {path}")
            continue

        mask = color_segmentation(img)
        boxes = find_bounding_boxes(mask)

        # Save the mask
        mask_filename = os.path.basename(path).replace('.jp2', '.png')
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)
        
        # Save bounding box data
        for (x, y, w, h) in boxes:
            all_boxes_data.append({
                'image_path': os.path.basename(path),
                'label': 'candida',
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h
            })

        print(f"Processed {os.path.basename(path)}: found {len(boxes)} regions.")

    # Save all bounding boxes to a single CSV
    df = pd.DataFrame(all_boxes_data)
    df.to_csv(os.path.join(annotations_dir, 'bounding_boxes.csv'), index=False)
    print(f"Saved all bounding box data to annotations/bounding_boxes.csv")

if __name__ == "__main__":
    try:
        config = load_config()
        data_root = config['paths']['data_root']
        
        # Create output directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        output_dir = os.path.join(project_root, 'results', 'automated_detection')
        masks_dir = os.path.join(project_root, 'results', 'segmentation_masks')
        
        print("Starting automated Candida detection via color-based segmentation...")
        all_images = get_image_paths(data_root)
        print(f"Found {len(all_images)} images to process")
        
        # Process all images
        all_results = process_and_save(all_images, output_dir, masks_dir)
        
        if all_results:
            print(f"\n=== DETECTION SUMMARY ===")
            print(f"Total detections: {len(all_results)}")
            print(f"Images processed: {len(all_images)}")
            print(f"Average detections per image: {len(all_results)/len(all_images):.2f}")
            
            # Show sample result
            print("\nGenerating sample visualization...")
            sample_image = all_images[0]
            sample_results = [r for r in all_results if r['image_path'] == os.path.basename(sample_image)]
            
            # Load corresponding mask
            mask_filename = os.path.splitext(os.path.basename(sample_image))[0] + '.png'
            mask_path = os.path.join(masks_dir, mask_filename)
            sample_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
            
            visualize_detection_results(sample_image, sample_results, sample_mask)
            
        print("Automated detection complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()