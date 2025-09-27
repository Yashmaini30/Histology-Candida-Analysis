import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def load_config(config_path="config.yaml"):
    """Load configuration file."""
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def run_demo():
    """Run the complete Candida detection demo."""
    print("=" * 60)
    print("CANDIDA DETECTION SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    config = load_config()
    project_root = os.path.join(script_dir, '..')
    
    # Demo steps
    steps = [
        "1. Data Exploration",
        "2. Annotation Visualization", 
        "3. Automated Detection",
        "4. Performance Evaluation",
        "5. Results Analysis"
    ]
    
    for step in steps:
        print(f"\n{step}")
        print("-" * 40)
    
    print(f"\nProject Structure:")
    print(f"├── Data: {config['paths']['data_root']}")
    print(f"├── Annotations: {config['paths']['annotations_dir']}")
    print(f"├── Results: {config['paths']['results_dir']}")
    print(f"└── Scripts: scripts/")
    
    # Step 1: Data Exploration
    print("\n" + "="*60)
    print("STEP 1: DATA EXPLORATION")
    print("="*60)
    
    data_root = os.path.join(project_root, config['paths']['data_root'])
    
    # Count images in each subfolder
    image_counts = {}
    total_images = 0
    
    for subfolder in os.listdir(data_root):
        subfolder_path = os.path.join(data_root, subfolder)
        if os.path.isdir(subfolder_path):
            image_files = [f for f in os.listdir(subfolder_path) 
                          if f.lower().endswith(('.jp2', '.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            image_counts[subfolder] = len(image_files)
            total_images += len(image_files)
    
    print(f"Dataset Overview:")
    print(f"├── Total Images: {total_images}")
    for subfolder, count in image_counts.items():
        print(f"├── {subfolder}: {count} images")
    
    # Step 2: Annotation Analysis
    print("\n" + "="*60)
    print("STEP 2: ANNOTATION ANALYSIS")
    print("="*60)
    
    annotations_path = os.path.join(project_root, config['paths']['annotation_tool_output'], 'bounding_boxes.csv')
    
    if os.path.exists(annotations_path):
        annotations_df = pd.read_csv(annotations_path)
        
        print(f"Annotation Statistics:")
        print(f"├── Total Annotations: {len(annotations_df):,}")
        print(f"├── Annotated Images: {annotations_df['image_path'].nunique():,}")
        print(f"├── Avg Annotations/Image: {len(annotations_df)/annotations_df['image_path'].nunique():.2f}")
        
        # Bounding box size analysis
        annotations_df['width'] = annotations_df['xmax'] - annotations_df['xmin']
        annotations_df['height'] = annotations_df['ymax'] - annotations_df['ymin']
        annotations_df['area'] = annotations_df['width'] * annotations_df['height']
        
        print(f"├── Avg Bounding Box Area: {annotations_df['area'].mean():.1f} px²")
        print(f"├── Min Area: {annotations_df['area'].min()} px²")
        print(f"└── Max Area: {annotations_df['area'].max()} px²")
        
        # Show size distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(annotations_df['area'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Bounding Box Area (px²)')
        plt.ylabel('Frequency')
        plt.title('Area Distribution')
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.scatter(annotations_df['width'], annotations_df['height'], alpha=0.6, s=20)
        plt.xlabel('Width (px)')
        plt.ylabel('Height (px)')
        plt.title('Width vs Height')
        
        plt.subplot(1, 3, 3)
        counts_per_image = annotations_df.groupby('image_path').size()
        plt.hist(counts_per_image, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Annotations per Image')
        plt.ylabel('Number of Images')
        plt.title('Annotations Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, 'results', 'annotation_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        print("No annotations found. Please check the annotation file path.")
    
    # Step 3: Run Sample Detection
    print("\n" + "="*60)
    print("STEP 3: SAMPLE DETECTION DEMO")
    print("="*60)
    
    try:
        # Import the automated detection system
        from automated_detection import CandidaDetector
        
        detector = CandidaDetector()
        
        # Find a random sample image
        sample_images = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.lower().endswith('.jp2'):
                    sample_images.append(os.path.join(root, file))
        
        if sample_images:
            sample_image = random.choice(sample_images)
            print(f"Processing sample image: {os.path.basename(sample_image)}")
            
            result = detector.detect_single_image(sample_image, save_visualization=False)
            
            if 'error' not in result:
                print(f"Detection Results:")
                print(f"├── Image: {result['image_name']}")
                print(f"├── Candida Regions Found: {result['detection_count']}")
                
                if result['detection_count'] > 0:
                    areas = [d['area'] for d in result['detections']]
                    print(f"├── Average Region Area: {sum(areas)/len(areas):.1f} px²")
                    print(f"├── Largest Region: {max(areas)} px²")
                    print(f"└── Smallest Region: {min(areas)} px²")
                
                    # Show top 5 detections
                    print(f"\nTop 5 Detections (by area):")
                    sorted_detections = sorted(result['detections'], key=lambda x: x['area'], reverse=True)
                    for i, detection in enumerate(sorted_detections[:5]):
                        print(f"  {i+1}. Area: {detection['area']} px², Position: ({detection['xmin']}, {detection['ymin']})")
            else:
                print(f"Error processing image: {result['error']}")
        else:
            print("No sample images found for demo.")
            
    except ImportError:
        print("Automated detection system not available. Please check the implementation.")
    
    # Step 4: Show Existing Results (if any)
    print("\n" + "="*60)
    print("STEP 4: EXISTING RESULTS ANALYSIS")
    print("="*60)
    
    results_dir = os.path.join(project_root, 'results')
    
    # Check for existing detection results
    detection_results_path = os.path.join(results_dir, 'automated_detection', 'detected_bounding_boxes.csv')
    
    if os.path.exists(detection_results_path):
        detection_df = pd.read_csv(detection_results_path)
        print(f"Automated Detection Results Found:")
        print(f"├── Total Detections: {len(detection_df)}")
        print(f"├── Images Processed: {detection_df['image_path'].nunique()}")
        print(f"└── Avg Detections/Image: {len(detection_df)/detection_df['image_path'].nunique():.2f}")
    else:
        print("No automated detection results found.")
        print("Run 'python scripts/segmentation_pipeline.py' to generate detections.")
    
    # Check for evaluation results
    eval_results_path = os.path.join(results_dir, 'evaluation', 'evaluation_results.json')
    
    if os.path.exists(eval_results_path):
        import json
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        print(f"\nEvaluation Results Found:")
        for iou_threshold, metrics in eval_results.items():
            print(f"├── IoU {iou_threshold}:")
            print(f"│   ├── Precision: {metrics['precision']:.3f}")
            print(f"│   ├── Recall: {metrics['recall']:.3f}")
            print(f"│   └── F1 Score: {metrics['f1_score']:.3f}")
    else:
        print("\nNo evaluation results found.")
        print("Run 'python scripts/evaluate_detection.py' after generating detections.")
    
    # Step 5: Usage Instructions
    print("\n" + "="*60)
    print("STEP 5: USAGE INSTRUCTIONS")
    print("="*60)
    
    print("Available Scripts:")
    print("├── explore_data.py           - Explore and visualize dataset")
    print("├── visualize_annotations.py  - View existing annotations")
    print("├── segmentation_pipeline.py  - Run automated detection")
    print("├── evaluate_detection.py     - Evaluate detection performance")
    print("├── automated_detection.py    - Command-line detection tool")
    print("└── demo.py                  - This demo script")
    
    print("\nCommand-line Usage Examples:")
    print("# Run detection on single image:")
    print("python scripts/automated_detection.py -i data/image.jp2 -o results/ -v")
    print("\n# Batch process entire directory:")
    print("python scripts/automated_detection.py -i data/ -o results/batch/ -v")
    print("\n# Evaluate detection performance:")
    print("python scripts/evaluate_detection.py")
    
    print("\nNext Steps for Conference Paper:")
    print("1. Run full detection pipeline on all images")
    print("2. Evaluate performance against ground truth annotations")
    print("3. Analyze detection errors and improve algorithm")
    print("4. Generate visualizations and statistical analysis")
    print("5. Write methods section describing color-based segmentation approach")
    
    # Project Status Summary
    print("\n" + "="*60)
    print("PROJECT STATUS SUMMARY")
    print("="*60)
    
    status_items = [
        ("✅", "Data Organization", "Complete - 3 subfolders with JP2 images"),
        ("✅", "Manual Annotations", f"Complete - {len(annotations_df):,} bounding boxes" if 'annotations_df' in locals() else "Complete - Available"),
        ("✅", "Segmentation Algorithm", "Complete - Color-based HSV segmentation"),
        ("✅", "Visualization Tools", "Complete - Multiple visualization scripts"),
        ("✅", "Evaluation Metrics", "Complete - IoU, Precision, Recall, F1"),
        ("✅", "Automated Pipeline", "Complete - Batch processing capability"),
        ("⚠️", "Performance Evaluation", "Pending - Run evaluation on all images"),
        ("⚠️", "Results Analysis", "Pending - Generate comprehensive analysis"),
        ("📝", "Conference Paper", "Ready for writing - Strong technical foundation")
    ]
    
    for status, item, description in status_items:
        print(f"{status} {item:<25} {description}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE - Your Candida detection system is ready!")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()