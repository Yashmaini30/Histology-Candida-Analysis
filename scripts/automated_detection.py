import os
import cv2
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import functions from other scripts
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def color_segmentation(image):
    """
    Performs color-based segmentation to find dark, reddish/purplish regions.
    Returns a binary mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV range for Candida (dark purple/reddish regions)
    lower_purple = np.array([130, 100, 20])
    upper_purple = np.array([160, 255, 150])
    
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Post-processing to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def find_bounding_boxes(mask, min_area=50, min_solidity=0.5):
    """Finds bounding boxes for segmented regions with filtering."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Calculate solidity as a shape filter
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > min_solidity:
                    x, y, w, h = cv2.boundingRect(contour)
                    filtered_boxes.append((x, y, w, h))
    
    return filtered_boxes

class CandidaDetector:
    """
    Main Candida detection class that encapsulates the detection pipeline.
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.detection_params = {
            'min_area': 50,
            'min_solidity': 0.5,
            'confidence_threshold': 0.7
        }
        
    def detect_single_image(self, image_path, save_visualization=False, output_dir=None):
        """
        Detect Candida organisms in a single image.
        
        Args:
            image_path: Path to the input image
            save_visualization: Whether to save visualization images
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Perform segmentation
            mask = color_segmentation(image)
            
            # Find bounding boxes
            boxes = find_bounding_boxes(
                mask, 
                min_area=self.detection_params['min_area'],
                min_solidity=self.detection_params['min_solidity']
            )
            
            # Prepare results
            image_name = os.path.basename(image_path)
            detections = []
            
            for i, (x, y, w, h) in enumerate(boxes):
                detections.append({
                    'detection_id': i + 1,
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + w,
                    'ymax': y + h,
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'confidence': 0.95  # High confidence for rule-based detection
                })
            
            results = {
                'image_path': image_path,
                'image_name': image_name,
                'detection_count': len(detections),
                'detections': detections,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Save visualization if requested
            if save_visualization and output_dir:
                self._save_visualization(image, mask, boxes, image_name, output_dir)
            
            return results
            
        except Exception as e:
            return {
                'image_path': image_path,
                'image_name': os.path.basename(image_path) if image_path else 'unknown',
                'detection_count': 0,
                'detections': [],
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def detect_batch(self, input_dir, output_dir=None, save_visualizations=False):
        """
        Detect Candida organisms in all images within a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            save_visualizations: Whether to save visualization images
            
        Returns:
            List of detection results for all images
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ('.jp2', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_paths = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images to process...")
        
        all_results = []
        successful_detections = 0
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            result = self.detect_single_image(
                image_path, 
                save_visualization=save_visualizations,
                output_dir=output_dir
            )
            
            all_results.append(result)
            
            if 'error' not in result:
                successful_detections += 1
                print(f"  -> Found {result['detection_count']} Candida regions")
            else:
                print(f"  -> Error: {result['error']}")
        
        print(f"\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Total images processed: {len(image_paths)}")
        print(f"Successful detections: {successful_detections}")
        print(f"Failed processing: {len(image_paths) - successful_detections}")
        
        total_detections = sum(r['detection_count'] for r in all_results)
        print(f"Total Candida regions detected: {total_detections}")
        print(f"Average detections per image: {total_detections/successful_detections:.2f}")
        
        # Save results
        if output_dir:
            self._save_batch_results(all_results, output_dir)
        
        return all_results
    
    def _save_visualization(self, image, mask, boxes, image_name, output_dir):
        """Save visualization of detection results."""
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Image with detections
        image_with_boxes = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f'C{i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        axes[2].imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Detections ({len(boxes)} regions)')
        axes[2].axis('off')
        
        # Save visualization
        viz_name = os.path.splitext(image_name)[0] + '_detection.png'
        viz_path = os.path.join(viz_dir, viz_name)
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_batch_results(self, results, output_dir):
        """Save batch processing results to CSV and JSON files."""
        # Prepare data for CSV (flattened detection results)
        csv_data = []
        summary_data = []
        
        for result in results:
            image_info = {
                'image_name': result['image_name'],
                'image_path': result['image_path'],
                'detection_count': result['detection_count'],
                'processing_timestamp': result['processing_timestamp']
            }
            
            if 'error' in result:
                image_info['error'] = result['error']
                summary_data.append(image_info)
            else:
                summary_data.append(image_info)
                
                # Add individual detections to CSV data
                for detection in result['detections']:
                    csv_row = {
                        'image_name': result['image_name'],
                        'detection_id': detection['detection_id'],
                        'xmin': detection['xmin'],
                        'ymin': detection['ymin'],
                        'xmax': detection['xmax'],
                        'ymax': detection['ymax'],
                        'width': detection['width'],
                        'height': detection['height'],
                        'area': detection['area'],
                        'confidence': detection['confidence']
                    }
                    csv_data.append(csv_row)
        
        # Save detailed detections to CSV
        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, 'candida_detections.csv')
            csv_df.to_csv(csv_path, index=False)
            print(f"Detailed detections saved to: {csv_path}")
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'detection_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Detection summary saved to: {summary_path}")
        
        # Save full results to JSON
        import json
        json_path = os.path.join(output_dir, 'full_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Full results saved to: {json_path}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Automated Candida Detection System")
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or directory')
    parser.add_argument('--output', '-o', 
                       help='Output directory for results')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CandidaDetector(args.config)
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        print(f"Processing single image: {args.input}")
        result = detector.detect_single_image(
            args.input, 
            save_visualization=args.visualize,
            output_dir=args.output
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Found {result['detection_count']} Candida regions")
            for i, detection in enumerate(result['detections']):
                print(f"  Detection {i+1}: Area={detection['area']} px², "
                      f"Position=({detection['xmin']},{detection['ymin']})")
    
    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        results = detector.detect_batch(
            args.input,
            output_dir=args.output,
            save_visualizations=args.visualize
        )
    
    else:
        print(f"Error: Input path '{args.input}' is not a valid file or directory")
        return 1
    
    print("Detection complete!")
    return 0

if __name__ == "__main__":
    exit(main())