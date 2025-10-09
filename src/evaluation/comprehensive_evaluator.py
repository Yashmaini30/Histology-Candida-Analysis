"""
Comprehensive Evaluation Framework for Candida Detection System
Implements standard computer vision metrics for conference paper validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import os
from collections import defaultdict
import json

class DetectionEvaluator:
    """
    Comprehensive evaluation of object detection performance
    """
    
    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            self.iou_thresholds = [0.3, 0.5, 0.7, 0.9]
        else:
            self.iou_thresholds = iou_thresholds
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        box format: [xmin, ymin, xmax, ymax]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def match_detections(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        Match predicted boxes to ground truth boxes based on IoU threshold
        """
        if len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), []
        
        if len(pred_boxes) == 0:
            return [], [], list(range(len(gt_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Hungarian algorithm approximation (greedy matching)
        matched_pairs = []
        unmatched_pred = list(range(len(pred_boxes)))
        unmatched_gt = list(range(len(gt_boxes)))
        
        # Sort by highest IoU first
        pred_gt_pairs = []
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    pred_gt_pairs.append((i, j, iou_matrix[i, j]))
        
        pred_gt_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Match pairs greedily
        used_pred = set()
        used_gt = set()
        
        for pred_idx, gt_idx, iou_score in pred_gt_pairs:
            if pred_idx not in used_pred and gt_idx not in used_gt:
                matched_pairs.append((pred_idx, gt_idx, iou_score))
                used_pred.add(pred_idx)
                used_gt.add(gt_idx)
        
        # Update unmatched lists
        unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
        
        return matched_pairs, unmatched_pred, unmatched_gt
    
    def calculate_precision_recall(self, pred_boxes, gt_boxes, confidence_scores=None):
        """
        Calculate precision and recall at different confidence thresholds
        """
        if confidence_scores is None:
            confidence_scores = [0.95] * len(pred_boxes)
        
        # Sort predictions by confidence (descending)
        sorted_indices = sorted(range(len(confidence_scores)), 
                              key=lambda i: confidence_scores[i], reverse=True)
        
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            precisions = []
            recalls = []
            confidences = []
            
            # Vary confidence threshold
            for conf_thresh in np.arange(0.1, 1.0, 0.1):
                # Filter predictions by confidence
                filtered_pred_boxes = []
                for idx in sorted_indices:
                    if confidence_scores[idx] >= conf_thresh:
                        filtered_pred_boxes.append(pred_boxes[idx])
                
                if len(filtered_pred_boxes) == 0:
                    precision = 0.0 if len(gt_boxes) > 0 else 1.0
                    recall = 0.0
                else:
                    # Match detections
                    matches, unmatched_pred, unmatched_gt = self.match_detections(
                        filtered_pred_boxes, gt_boxes, iou_thresh
                    )
                    
                    tp = len(matches)
                    fp = len(unmatched_pred)
                    fn = len(unmatched_gt)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                confidences.append(conf_thresh)
            
            results[f'iou_{iou_thresh}'] = {
                'precisions': precisions,
                'recalls': recalls,
                'confidences': confidences
            }
        
        return results
    
    def calculate_ap(self, precisions, recalls):
        """
        Calculate Average Precision (AP) using the 11-point interpolation method
        """
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[sorted_indices]
        precisions_sorted = np.array(precisions)[sorted_indices]
        
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions for recall >= t
            precs = precisions_sorted[recalls_sorted >= t]
            if len(precs) > 0:
                ap += np.max(precs) / 11.0
        
        return ap

class SegmentationEvaluator:
    """
    Evaluation metrics for semantic segmentation
    """
    
    @staticmethod
    def calculate_dice_coefficient(pred_mask, gt_mask):
        """Calculate Dice coefficient (F1 score for segmentation)"""
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        total = np.sum(pred_mask) + np.sum(gt_mask)
        
        return (2.0 * intersection) / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_iou_segmentation(pred_mask, gt_mask):
        """Calculate IoU for segmentation masks"""
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum((pred_mask + gt_mask) > 0)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_pixel_accuracy(pred_mask, gt_mask):
        """Calculate pixel-wise accuracy"""
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        correct_pixels = np.sum(pred_mask == gt_mask)
        total_pixels = pred_mask.size
        
        return correct_pixels / total_pixels

class ComprehensiveEvaluator:
    """
    Main evaluation class combining detection and segmentation metrics
    """
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.detection_evaluator = DetectionEvaluator()
        self.segmentation_evaluator = SegmentationEvaluator()
        self.results = {}
    
    def load_detection_results(self, detection_csv):
        """Load detection results from CSV"""
        return pd.read_csv(detection_csv)
    
    def create_synthetic_ground_truth(self, detection_df):
        """
        Create synthetic ground truth for evaluation
        (In real scenario, this would be manually annotated data)
        """
        # For demonstration, we'll create ground truth based on detection results
        # with some noise to simulate real annotation differences
        
        gt_data = []
        
        for _, row in detection_df.iterrows():
            # Add some noise to simulate annotation differences
            noise_factor = 0.1
            width = row['xmax'] - row['xmin']
            height = row['ymax'] - row['ymin']
            
            noise_x = np.random.uniform(-noise_factor * width, noise_factor * width)
            noise_y = np.random.uniform(-noise_factor * height, noise_factor * height)
            noise_w = np.random.uniform(-noise_factor * width, noise_factor * width)
            noise_h = np.random.uniform(-noise_factor * height, noise_factor * height)
            
            # Only include ~90% of detections as ground truth (simulate missed annotations)
            if np.random.random() > 0.1:
                gt_data.append({
                    'image_name': row['image_name'],
                    'xmin': max(0, row['xmin'] + noise_x),
                    'ymin': max(0, row['ymin'] + noise_y),
                    'xmax': row['xmax'] + noise_w,
                    'ymax': row['ymax'] + noise_h,
                    'label': 'candida'
                })
        
        return pd.DataFrame(gt_data)
    
    def evaluate_detection_performance(self, predictions_df, ground_truth_df):
        """
        Comprehensive evaluation of detection performance
        """
        results = defaultdict(list)
        image_results = {}
        
        # Group by image
        for image_name in predictions_df['image_name'].unique():
            pred_image = predictions_df[predictions_df['image_name'] == image_name]
            gt_image = ground_truth_df[ground_truth_df['image_name'] == image_name]
            
            # Convert to box format
            pred_boxes = []
            for _, row in pred_image.iterrows():
                pred_boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            
            gt_boxes = []
            for _, row in gt_image.iterrows():
                gt_boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            
            # Calculate metrics for each IoU threshold
            image_metrics = {}
            for iou_thresh in self.detection_evaluator.iou_thresholds:
                matches, unmatched_pred, unmatched_gt = self.detection_evaluator.match_detections(
                    pred_boxes, gt_boxes, iou_thresh
                )
                
                tp = len(matches)
                fp = len(unmatched_pred)
                fn = len(unmatched_gt)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                image_metrics[f'iou_{iou_thresh}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'tp': tp, 'fp': fp, 'fn': fn
                }
            
            image_results[image_name] = image_metrics
        
        # Calculate overall metrics
        overall_metrics = {}
        for iou_thresh in self.detection_evaluator.iou_thresholds:
            total_tp = sum([img[f'iou_{iou_thresh}']['tp'] for img in image_results.values()])
            total_fp = sum([img[f'iou_{iou_thresh}']['fp'] for img in image_results.values()])
            total_fn = sum([img[f'iou_{iou_thresh}']['fn'] for img in image_results.values()])
            
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            
            overall_metrics[f'iou_{iou_thresh}'] = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            }
        
        return overall_metrics, image_results
    
    def create_evaluation_report(self, overall_metrics, output_dir):
        """
        Create comprehensive evaluation report with visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics summary table
        summary_data = []
        for iou_key, metrics in overall_metrics.items():
            iou_value = float(iou_key.split('_')[1])
            summary_data.append({
                'IoU_Threshold': iou_value,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'True_Positives': metrics['total_tp'],
                'False_Positives': metrics['total_fp'],
                'False_Negatives': metrics['total_fn']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'evaluation_summary.csv'), index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision vs IoU threshold
        axes[0,0].plot(summary_df['IoU_Threshold'], summary_df['Precision'], 'b-o', linewidth=2)
        axes[0,0].set_xlabel('IoU Threshold')
        axes[0,0].set_ylabel('Precision')
        axes[0,0].set_title('Precision vs IoU Threshold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim([0, 1])
        
        # Recall vs IoU threshold
        axes[0,1].plot(summary_df['IoU_Threshold'], summary_df['Recall'], 'r-o', linewidth=2)
        axes[0,1].set_xlabel('IoU Threshold')
        axes[0,1].set_ylabel('Recall')
        axes[0,1].set_title('Recall vs IoU Threshold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim([0, 1])
        
        # F1 Score vs IoU threshold
        axes[1,0].plot(summary_df['IoU_Threshold'], summary_df['F1_Score'], 'g-o', linewidth=2)
        axes[1,0].set_xlabel('IoU Threshold')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].set_title('F1 Score vs IoU Threshold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim([0, 1])
        
        # Precision-Recall curve
        axes[1,1].plot(summary_df['Recall'], summary_df['Precision'], 'purple', linewidth=2, marker='o')
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title('Precision-Recall Curve')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim([0, 1])
        axes[1,1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrix visualization for IoU=0.5
        iou_05_metrics = overall_metrics['iou_0.5']
        cm_data = np.array([[iou_05_metrics['total_tp'], iou_05_metrics['total_fn']],
                           [iou_05_metrics['total_fp'], 0]])  # TN not applicable in object detection
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Positive', 'Negative'],
                   yticklabels=['True', 'False'])
        plt.title('Detection Results at IoU=0.5')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return summary_df

def main():
    """
    Main evaluation function
    """
    # Paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    detection_csv = os.path.join(project_root, 'results', 'detection', 'candida_detections.csv')
    output_dir = os.path.join(project_root, 'results', 'evaluation')
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(project_root)
    
    # Load detection results
    print("Loading detection results...")
    detection_df = evaluator.load_detection_results(detection_csv)
    print(f"Loaded {len(detection_df)} detections")
    
    # Create synthetic ground truth (replace with real annotations for actual evaluation)
    print("Creating synthetic ground truth...")
    gt_df = evaluator.create_synthetic_ground_truth(detection_df)
    print(f"Created {len(gt_df)} ground truth annotations")
    
    # Evaluate detection performance
    print("Evaluating detection performance...")
    overall_metrics, image_results = evaluator.evaluate_detection_performance(detection_df, gt_df)
    
    # Create evaluation report
    print("Creating evaluation report...")
    summary_df = evaluator.create_evaluation_report(overall_metrics, output_dir)
    
    # Print summary
    print("\nEVALUATION SUMMARY:")
    print("=" * 50)
    for _, row in summary_df.iterrows():
        iou = row['IoU_Threshold']
        prec = row['Precision']
        rec = row['Recall']
        f1 = row['F1_Score']
        print(f"IoU={iou:.1f}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    
    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    main()