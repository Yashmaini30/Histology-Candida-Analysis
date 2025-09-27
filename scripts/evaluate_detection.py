import os
import cv2
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
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

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    box format: (xmin, ymin, xmax, ymax)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    intersection_xmin = max(x1_min, x2_min)
    intersection_ymin = max(y1_min, y2_min)
    intersection_xmax = min(x1_max, x2_max)
    intersection_ymax = min(y1_max, y2_max)
    
    if intersection_xmax <= intersection_xmin or intersection_ymax <= intersection_ymin:
        return 0.0
    
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def match_detections(ground_truth_boxes, detected_boxes, iou_threshold=0.5):
    """
    Match detected boxes with ground truth boxes using IoU threshold.
    Returns matches, false positives, and false negatives.
    """
    gt_boxes = [(row['xmin'], row['ymin'], row['xmax'], row['ymax']) for _, row in ground_truth_boxes.iterrows()]
    det_boxes = [(row['xmin'], row['ymin'], row['xmax'], row['ymax']) for _, row in detected_boxes.iterrows()]
    
    matched_gt = set()
    matched_det = set()
    matches = []
    
    # Find matches
    for i, det_box in enumerate(det_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
                
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j
        
        if best_gt_idx != -1:
            matches.append((i, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
            matched_det.add(i)
    
    # False positives: detected boxes that don't match any ground truth
    false_positives = [i for i in range(len(det_boxes)) if i not in matched_det]
    
    # False negatives: ground truth boxes that don't match any detection
    false_negatives = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    
    return matches, false_positives, false_negatives

def evaluate_image(image_name, ground_truth_df, detected_df, iou_threshold=0.5):
    """
    Evaluate detection performance on a single image.
    """
    # Get boxes for this image
    gt_boxes = ground_truth_df[ground_truth_df['image_path'] == image_name]
    det_boxes = detected_df[detected_df['image_path'] == image_name]
    
    if len(gt_boxes) == 0 and len(det_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'matches': []}
    
    if len(det_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_boxes), 'matches': []}
    
    if len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(det_boxes), 'fn': 0, 'matches': []}
    
    # Match detections
    matches, false_positives, false_negatives = match_detections(gt_boxes, det_boxes, iou_threshold)
    
    return {
        'tp': len(matches),
        'fp': len(false_positives),
        'fn': len(false_negatives),
        'matches': matches
    }

def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def evaluate_detection_performance(ground_truth_csv, detected_csv, iou_thresholds=[0.3, 0.5, 0.7]):
    """
    Comprehensive evaluation of detection performance.
    """
    print("Loading annotation files...")
    ground_truth_df = pd.read_csv(ground_truth_csv)
    detected_df = pd.read_csv(detected_csv)
    
    print(f"Ground truth: {len(ground_truth_df)} annotations in {ground_truth_df['image_path'].nunique()} images")
    print(f"Detections: {len(detected_df)} annotations in {detected_df['image_path'].nunique()} images")
    
    results = {}
    
    for iou_threshold in iou_thresholds:
        print(f"\n=== Evaluating at IoU threshold {iou_threshold} ===")
        
        total_tp = total_fp = total_fn = 0
        image_results = []
        
        # Get common images (images that appear in both datasets)
        common_images = set(ground_truth_df['image_path'].unique()) & set(detected_df['image_path'].unique())
        print(f"Evaluating {len(common_images)} common images...")
        
        for image_name in common_images:
            result = evaluate_image(image_name, ground_truth_df, detected_df, iou_threshold)
            image_results.append({
                'image': image_name,
                'tp': result['tp'],
                'fp': result['fp'],
                'fn': result['fn']
            })
            
            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
        
        # Calculate overall metrics
        precision, recall, f1_score = calculate_metrics(total_tp, total_fp, total_fn)
        
        results[iou_threshold] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'image_results': image_results
        }
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")
        print(f"True Positives: {total_tp}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
    
    return results

def plot_evaluation_results(results):
    """Plot evaluation metrics across different IoU thresholds."""
    iou_thresholds = list(results.keys())
    precisions = [results[iou]['precision'] for iou in iou_thresholds]
    recalls = [results[iou]['recall'] for iou in iou_thresholds]
    f1_scores = [results[iou]['f1_score'] for iou in iou_thresholds]
    
    plt.figure(figsize=(15, 5))
    
    # Precision-Recall curve
    plt.subplot(1, 3, 1)
    plt.plot(iou_thresholds, precisions, 'b-o', label='Precision')
    plt.plot(iou_thresholds, recalls, 'r-o', label='Recall')
    plt.plot(iou_thresholds, f1_scores, 'g-o', label='F1 Score')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs IoU Threshold')
    plt.legend()
    plt.grid(True)
    
    # Bar chart of TP/FP/FN at IoU=0.5
    if 0.5 in results:
        result_05 = results[0.5]
        plt.subplot(1, 3, 2)
        categories = ['True Positives', 'False Positives', 'False Negatives']
        values = [result_05['tp'], result_05['fp'], result_05['fn']]
        colors = ['green', 'orange', 'red']
        plt.bar(categories, values, color=colors)
        plt.title('Detection Results (IoU=0.5)')
        plt.ylabel('Count')
        for i, v in enumerate(values):
            plt.text(i, v + max(values)*0.01, str(v), ha='center', va='bottom')
    
    # Performance summary
    plt.subplot(1, 3, 3)
    plt.axis('off')
    summary_text = "Detection Performance Summary\n\n"
    for iou in iou_thresholds:
        result = results[iou]
        summary_text += f"IoU {iou}:\n"
        summary_text += f"  Precision: {result['precision']:.3f}\n"
        summary_text += f"  Recall: {result['recall']:.3f}\n"
        summary_text += f"  F1: {result['f1_score']:.3f}\n\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def analyze_detection_errors(ground_truth_csv, detected_csv, output_dir=None):
    """
    Analyze common detection errors and provide insights.
    """
    ground_truth_df = pd.read_csv(ground_truth_csv)
    detected_df = pd.read_csv(detected_csv)
    
    # Size analysis
    print("=== SIZE ANALYSIS ===")
    
    # Ground truth sizes
    gt_widths = ground_truth_df['xmax'] - ground_truth_df['xmin']
    gt_heights = ground_truth_df['ymax'] - ground_truth_df['ymin']
    gt_areas = gt_widths * gt_heights
    
    # Detection sizes
    det_widths = detected_df['xmax'] - detected_df['xmin']
    det_heights = detected_df['ymax'] - detected_df['ymin']
    det_areas = det_widths * det_heights
    
    print(f"Ground Truth - Avg area: {gt_areas.mean():.1f}, Std: {gt_areas.std():.1f}")
    print(f"Detections - Avg area: {det_areas.mean():.1f}, Std: {det_areas.std():.1f}")
    
    # Plot size distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(gt_areas, bins=50, alpha=0.7, label='Ground Truth', color='blue')
    plt.hist(det_areas, bins=50, alpha=0.7, label='Detections', color='red')
    plt.xlabel('Bounding Box Area (pixels²)')
    plt.ylabel('Frequency')
    plt.title('Size Distribution Comparison')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.scatter(gt_widths, gt_heights, alpha=0.6, s=20, label='Ground Truth')
    plt.scatter(det_widths, det_heights, alpha=0.6, s=20, label='Detections')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Width vs Height')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Detection count per image
    gt_counts = ground_truth_df.groupby('image_path').size()
    det_counts = detected_df.groupby('image_path').size()
    
    common_images = set(gt_counts.index) & set(det_counts.index)
    gt_common = [gt_counts.get(img, 0) for img in common_images]
    det_common = [det_counts.get(img, 0) for img in common_images]
    
    plt.scatter(gt_common, det_common, alpha=0.6)
    plt.xlabel('Ground Truth Count per Image')
    plt.ylabel('Detection Count per Image')
    plt.title('Detections vs Ground Truth Count')
    
    # Add perfect correlation line
    max_count = max(max(gt_common), max(det_common))
    plt.plot([0, max_count], [0, max_count], 'r--', alpha=0.8, label='Perfect Match')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'gt_area_stats': {'mean': gt_areas.mean(), 'std': gt_areas.std()},
        'det_area_stats': {'mean': det_areas.mean(), 'std': det_areas.std()},
        'common_images': len(common_images)
    }

if __name__ == "__main__":
    try:
        config = load_config()
        
        # Define paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        
        ground_truth_csv = os.path.join(project_root, config['paths']['annotation_tool_output'], 'bounding_boxes.csv')
        detected_csv = os.path.join(project_root, 'results', 'automated_detection', 'bounding_boxes.csv')
        
        # Check if detection results exist
        if not os.path.exists(detected_csv):
            print(f"Detection results not found at {detected_csv}")
            print("Please run segmentation_pipeline.py first to generate detections.")
            exit(1)
        
        print("=== CANDIDA DETECTION EVALUATION ===")
        
        # Perform evaluation
        results = evaluate_detection_performance(ground_truth_csv, detected_csv)
        
        # Plot results
        plot_evaluation_results(results)
        
        # Analyze errors
        print("\n=== ERROR ANALYSIS ===")
        error_analysis = analyze_detection_errors(ground_truth_csv, detected_csv)
        
        # Save results
        output_dir = os.path.join(project_root, 'results', 'evaluation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation metrics
        import json
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for iou, metrics in results.items():
                json_results[str(iou)] = {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                    'tp': int(metrics['tp']),
                    'fp': int(metrics['fp']),
                    'fn': int(metrics['fn'])
                }
            json.dump(json_results, f, indent=2)
        
        print(f"Evaluation results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()