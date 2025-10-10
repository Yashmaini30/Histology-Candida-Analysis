"""
Comparative Analysis Framework for Conference Paper
Compares different approaches: Color-based vs Deep Learning, Parameter tuning, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import time
from datetime import datetime
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from segmentation_pipeline import color_segmentation, find_filtered_bounding_boxes
from image_utils import load_image_robust

class ParameterSensitivityAnalyzer:
    """
    Analyze sensitivity of HSV parameters for color-based segmentation
    """
    
    def __init__(self):
        self.base_params = {
            'hue_range': [130, 160],
            'saturation_range': [100, 255], 
            'value_range': [20, 150]
        }
        
    def generate_parameter_variations(self):
        """Generate different HSV parameter combinations for testing"""
        variations = []
        
        # Hue variations
        hue_ranges = [[120, 150], [130, 160], [140, 170], [125, 165]]
        
        # Saturation variations  
        sat_ranges = [[80, 255], [100, 255], [120, 255], [90, 240]]
        
        # Value variations
        val_ranges = [[10, 140], [20, 150], [30, 160], [15, 130]]
        
        for i, (h_range, s_range, v_range) in enumerate(zip(hue_ranges, sat_ranges, val_ranges)):
            variations.append({
                'name': f'HSV_Variant_{i+1}',
                'hue_range': h_range,
                'saturation_range': s_range,
                'value_range': v_range
            })
        
        return variations
    
    def test_parameter_variation(self, image, params):
        """Test a single parameter variation on an image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_bound = np.array([params['hue_range'][0], params['saturation_range'][0], params['value_range'][0]])
        upper_bound = np.array([params['hue_range'][1], params['saturation_range'][1], params['value_range'][1]])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Post-processing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find bounding boxes
        boxes = find_filtered_bounding_boxes(mask)
        
        return {
            'detection_count': len(boxes),
            'mask_coverage': np.sum(mask > 0) / mask.size,
            'boxes': boxes
        }

class AlgorithmComparator:
    """
    Compare different segmentation algorithms
    """
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.algorithms = {
            'color_based_hsv': self.color_based_segmentation,
            'adaptive_threshold': self.adaptive_threshold_segmentation,
            'watershed': self.watershed_segmentation,
            'kmeans_clustering': self.kmeans_segmentation
        }
    
    def color_based_segmentation(self, image):
        """Original HSV color-based approach"""
        start_time = time.time()
        mask = color_segmentation(image)
        boxes = find_filtered_bounding_boxes(mask)
        processing_time = time.time() - start_time
        
        return {
            'mask': mask,
            'boxes': boxes,
            'detection_count': len(boxes),
            'processing_time': processing_time
        }
    
    def adaptive_threshold_segmentation(self, image):
        """Adaptive thresholding approach"""
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        # Post-processing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        boxes = find_filtered_bounding_boxes(mask)
        processing_time = time.time() - start_time
        
        return {
            'mask': mask,
            'boxes': boxes,
            'detection_count': len(boxes),
            'processing_time': processing_time
        }
    
    def watershed_segmentation(self, image):
        """Watershed-based segmentation"""
        start_time = time.time()
        
        # Convert to grayscale and apply initial processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(image, markers)
        
        # Create mask from watershed result
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == -1] = 255  # Boundaries
        mask[markers > 1] = 255    # Foreground regions
        
        boxes = find_filtered_bounding_boxes(mask)
        processing_time = time.time() - start_time
        
        return {
            'mask': mask,
            'boxes': boxes,
            'detection_count': len(boxes),
            'processing_time': processing_time
        }
    
    def kmeans_segmentation(self, image, k=3):
        """K-means clustering based segmentation"""
        start_time = time.time()
        
        # Reshape image for K-means
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Map each pixel to its cluster center
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        # Convert to grayscale and threshold to create mask
        gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        
        # Find the darkest cluster (likely Candida)
        unique_vals = np.unique(gray_segmented)
        darkest_val = np.min(unique_vals)
        
        # Create mask for darkest regions
        mask = np.zeros(gray_segmented.shape, dtype=np.uint8)
        mask[gray_segmented == darkest_val] = 255
        
        # Post-processing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        boxes = find_filtered_bounding_boxes(mask)
        processing_time = time.time() - start_time
        
        return {
            'mask': mask,
            'boxes': boxes,
            'detection_count': len(boxes),
            'processing_time': processing_time
        }

class ComparativeAnalysis:
    """
    Main class for comprehensive comparative analysis
    """
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.param_analyzer = ParameterSensitivityAnalyzer()
        self.algorithm_comparator = AlgorithmComparator(project_root)
        self.results = {}
    
    def run_parameter_sensitivity_analysis(self, sample_images, output_dir):
        """Run parameter sensitivity analysis"""
        print("Running parameter sensitivity analysis...")
        
        variations = self.param_analyzer.generate_parameter_variations()
        results = []
        
        for image_path in sample_images[:5]:  # Test on first 5 images
            image = load_image_robust(image_path)
            if image is None:
                continue
                
            image_name = os.path.basename(image_path)
            
            for variant in variations:
                result = self.param_analyzer.test_parameter_variation(image, variant)
                results.append({
                    'image_name': image_name,
                    'variant_name': variant['name'],
                    'hue_min': variant['hue_range'][0],
                    'hue_max': variant['hue_range'][1],
                    'sat_min': variant['saturation_range'][0],
                    'sat_max': variant['saturation_range'][1],
                    'val_min': variant['value_range'][0],
                    'val_max': variant['value_range'][1],
                    'detection_count': result['detection_count'],
                    'mask_coverage': result['mask_coverage']
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'parameter_sensitivity_results.csv'), index=False)
        
        # Create visualizations
        self._visualize_parameter_sensitivity(results_df, output_dir)
        
        return results_df
    
    def run_algorithm_comparison(self, sample_images, output_dir):
        """Compare different algorithms"""
        print("Running algorithm comparison...")
        
        results = []
        
        for image_path in sample_images[:10]:  # Test on first 10 images
            image = load_image_robust(image_path)
            if image is None:
                continue
                
            image_name = os.path.basename(image_path)
            
            for alg_name, alg_func in self.algorithm_comparator.algorithms.items():
                try:
                    result = alg_func(image)
                    results.append({
                        'image_name': image_name,
                        'algorithm': alg_name,
                        'detection_count': result['detection_count'],
                        'processing_time': result['processing_time'],
                        'mask_pixels': np.sum(result['mask'] > 0) if 'mask' in result else 0,
                        'mask_coverage': np.sum(result['mask'] > 0) / result['mask'].size if 'mask' in result else 0
                    })
                except Exception as e:
                    print(f"Error with {alg_name} on {image_name}: {e}")
                    results.append({
                        'image_name': image_name,
                        'algorithm': alg_name,
                        'detection_count': 0,
                        'processing_time': 0,
                        'mask_pixels': 0,
                        'mask_coverage': 0,
                        'error': str(e)
                    })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'algorithm_comparison_results.csv'), index=False)
        
        # Create visualizations
        self._visualize_algorithm_comparison(results_df, output_dir)
        
        return results_df
    
    def _visualize_parameter_sensitivity(self, results_df, output_dir):
        """Create parameter sensitivity visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Detection count by variant
        avg_detections = results_df.groupby('variant_name')['detection_count'].mean().reset_index()
        axes[0,0].bar(avg_detections['variant_name'], avg_detections['detection_count'])
        axes[0,0].set_title('Average Detection Count by HSV Variant')
        axes[0,0].set_ylabel('Detection Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Mask coverage by variant
        avg_coverage = results_df.groupby('variant_name')['mask_coverage'].mean().reset_index()
        axes[0,1].bar(avg_coverage['variant_name'], avg_coverage['mask_coverage'])
        axes[0,1].set_title('Average Mask Coverage by HSV Variant')
        axes[0,1].set_ylabel('Mask Coverage')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Hue range vs detection count
        axes[1,0].scatter(results_df['hue_min'], results_df['detection_count'], alpha=0.6, color='blue', label='Hue Min')
        axes[1,0].scatter(results_df['hue_max'], results_df['detection_count'], alpha=0.6, color='red', label='Hue Max')
        axes[1,0].set_xlabel('Hue Value')
        axes[1,0].set_ylabel('Detection Count')
        axes[1,0].set_title('Hue Range vs Detection Count')
        axes[1,0].legend()
        
        # Saturation range vs mask coverage
        axes[1,1].scatter(results_df['sat_min'], results_df['mask_coverage'], alpha=0.6, color='green', label='Sat Min')
        axes[1,1].scatter(results_df['sat_max'], results_df['mask_coverage'], alpha=0.6, color='orange', label='Sat Max')
        axes[1,1].set_xlabel('Saturation Value')
        axes[1,1].set_ylabel('Mask Coverage')
        axes[1,1].set_title('Saturation Range vs Mask Coverage')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_algorithm_comparison(self, results_df, output_dir):
        """Create algorithm comparison visualizations"""
        # Remove error rows for clean visualization
        clean_df = results_df[~results_df.isin(['error']).any(axis=1)]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average detection count by algorithm
        avg_detections = clean_df.groupby('algorithm')['detection_count'].mean().reset_index()
        axes[0,0].bar(avg_detections['algorithm'], avg_detections['detection_count'])
        axes[0,0].set_title('Average Detection Count by Algorithm')
        axes[0,0].set_ylabel('Detection Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Processing time by algorithm
        avg_time = clean_df.groupby('algorithm')['processing_time'].mean().reset_index()
        axes[0,1].bar(avg_time['algorithm'], avg_time['processing_time'])
        axes[0,1].set_title('Average Processing Time by Algorithm')
        axes[0,1].set_ylabel('Processing Time (seconds)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Detection count distribution
        algorithms = clean_df['algorithm'].unique()
        detection_data = [clean_df[clean_df['algorithm'] == alg]['detection_count'].values for alg in algorithms]
        axes[1,0].boxplot(detection_data, labels=algorithms)
        axes[1,0].set_title('Detection Count Distribution by Algorithm')
        axes[1,0].set_ylabel('Detection Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Efficiency plot (detections per second)
        clean_df['efficiency'] = clean_df['detection_count'] / clean_df['processing_time']
        efficiency_data = [clean_df[clean_df['algorithm'] == alg]['efficiency'].values for alg in algorithms]
        axes[1,1].boxplot(efficiency_data, labels=algorithms)
        axes[1,1].set_title('Algorithm Efficiency (Detections/Second)')
        axes[1,1].set_ylabel('Efficiency')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_analysis(self, results_df, output_dir):
        """Perform statistical significance tests"""
        print("Performing statistical analysis...")
        
        if 'algorithm' in results_df.columns:
            # ANOVA test for algorithm comparison
            algorithms = results_df['algorithm'].unique()
            detection_groups = [results_df[results_df['algorithm'] == alg]['detection_count'].values 
                              for alg in algorithms]
            
            # Remove empty groups
            detection_groups = [group for group in detection_groups if len(group) > 0]
            
            if len(detection_groups) > 1:
                try:
                    f_stat, p_value = stats.f_oneway(*detection_groups)
                    
                    # Pairwise t-tests
                    pairwise_results = []
                    for i, alg1 in enumerate(algorithms):
                        for j, alg2 in enumerate(algorithms[i+1:], i+1):
                            group1 = results_df[results_df['algorithm'] == alg1]['detection_count'].values
                            group2 = results_df[results_df['algorithm'] == alg2]['detection_count'].values
                            
                            if len(group1) > 0 and len(group2) > 0:
                                t_stat, t_p_value = stats.ttest_ind(group1, group2)
                                pairwise_results.append({
                                    'algorithm_1': alg1,
                                    'algorithm_2': alg2,
                                    't_statistic': t_stat,
                                    'p_value': t_p_value,
                                    'significant': t_p_value < 0.05
                                })
                    
                    # Save statistical results
                    stats_results = {
                        'anova_f_statistic': f_stat,
                        'anova_p_value': p_value,
                        'anova_significant': p_value < 0.05,
                        'pairwise_tests': pairwise_results
                    }
                    
                    with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
                        json.dump(stats_results, f, indent=2, default=str)
                    
                    return stats_results
                except Exception as e:
                    print(f"Error in statistical analysis: {e}")
                    return None
        
        return None
    
    def create_conference_summary_report(self, output_dir):
        """Create comprehensive conference-ready report"""
        print("Creating conference summary report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'project_overview': {
                'title': 'Automated Candida Detection in Histological Images',
                'objective': 'Compare color-based and advanced segmentation approaches',
                'dataset': 'PAS-stained oral tissue specimens (palate and tongue)',
                'evaluation_metrics': ['Detection Count', 'Processing Time', 'Mask Coverage', 'Statistical Significance']
            },
            'methodology': {
                'color_based_segmentation': 'HSV color space filtering with morphological post-processing',
                'comparative_algorithms': ['Adaptive Thresholding', 'Watershed', 'K-means Clustering'],
                'parameter_optimization': 'Systematic variation of HSV thresholds',
                'statistical_validation': 'ANOVA and pairwise t-tests for significance testing'
            }
        }
        
        # Load existing results if available
        try:
            param_results = pd.read_csv(os.path.join(output_dir, 'parameter_sensitivity_results.csv'))
            alg_results = pd.read_csv(os.path.join(output_dir, 'algorithm_comparison_results.csv'))
            
            report['results_summary'] = {
                'parameter_sensitivity': {
                    'total_variants_tested': len(param_results['variant_name'].unique()),
                    'images_tested': len(param_results['image_name'].unique()),
                    'best_variant': param_results.groupby('variant_name')['detection_count'].mean().idxmax(),
                    'detection_count_range': [
                        float(param_results['detection_count'].min()),
                        float(param_results['detection_count'].max())
                    ]
                },
                'algorithm_comparison': {
                    'algorithms_tested': len(alg_results['algorithm'].unique()),
                    'images_tested': len(alg_results['image_name'].unique()),
                    'best_algorithm': alg_results.groupby('algorithm')['detection_count'].mean().idxmax(),
                    'processing_time_range': [
                        float(alg_results['processing_time'].min()),
                        float(alg_results['processing_time'].max())
                    ]
                }
            }
        except Exception as e:
            print(f"Could not load results for summary: {e}")
        
        # Save report
        with open(os.path.join(output_dir, 'conference_summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

def main():
    """
    Main comparative analysis function
    """
    # Setup paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    data_root = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'results', 'comparative_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample images
    sample_images = []
    for root, dirs, files in os.walk(data_root):
        for file in files[:15]:  # Limit to first 15 images for analysis
            if file.lower().endswith('.jp2'):
                sample_images.append(os.path.join(root, file))
    
    if len(sample_images) == 0:
        print("No sample images found for analysis")
        return
    
    print(f"Found {len(sample_images)} sample images for analysis")
    
    # Initialize analysis
    analyzer = ComparativeAnalysis(project_root)
    
    # Run parameter sensitivity analysis
    param_results = analyzer.run_parameter_sensitivity_analysis(sample_images, output_dir)
    
    # Run algorithm comparison
    alg_results = analyzer.run_algorithm_comparison(sample_images, output_dir)
    
    # Perform statistical analysis
    stats_results = analyzer.perform_statistical_analysis(alg_results, output_dir)
    
    # Create conference report
    conference_report = analyzer.create_conference_summary_report(output_dir)
    
    print(f"\nComparative analysis completed!")
    print(f"Results saved to: {output_dir}")
    print(f"- Parameter sensitivity: {len(param_results)} data points")
    print(f"- Algorithm comparison: {len(alg_results)} data points") 
    print(f"- Statistical analysis: {'Completed' if stats_results else 'Failed'}")
    print(f"- Conference report: Generated")

if __name__ == "__main__":
    main()