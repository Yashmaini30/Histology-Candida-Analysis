"""
Morphological Classification System for Candida Organisms
Classifies detected regions into: Yeast cells, Pseudohyphae, True hyphae
"""

import numpy as np
import cv2
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import joblib

class MorphologicalFeatureExtractor:
    """
    Extract morphological features for Candida classification
    """
    
    @staticmethod
    def calculate_aspect_ratio(contour):
        """Calculate aspect ratio of bounding rectangle"""
        x, y, w, h = cv2.boundingRect(contour)
        return max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    
    @staticmethod
    def calculate_solidity(contour):
        """Calculate solidity (contour area / convex hull area)"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return area / hull_area if hull_area > 0 else 0.0
    
    @staticmethod
    def calculate_extent(contour):
        """Calculate extent (contour area / bounding rectangle area)"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        return area / rect_area if rect_area > 0 else 0.0
    
    @staticmethod
    def calculate_circularity(contour):
        """Calculate circularity (4π * area / perimeter²)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        return 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    
    @staticmethod
    def calculate_eccentricity(contour):
        """Calculate eccentricity using fitted ellipse"""
        if len(contour) < 5:
            return 0.0
        try:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis > 0:
                return np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except:
            return 0.0
        return 0.0
    
    @staticmethod
    def extract_features(contour):
        """Extract all morphological features for a contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        features = {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': MorphologicalFeatureExtractor.calculate_aspect_ratio(contour),
            'solidity': MorphologicalFeatureExtractor.calculate_solidity(contour),
            'extent': MorphologicalFeatureExtractor.calculate_extent(contour),
            'circularity': MorphologicalFeatureExtractor.calculate_circularity(contour),
            'eccentricity': MorphologicalFeatureExtractor.calculate_eccentricity(contour),
            'compactness': perimeter ** 2 / area if area > 0 else 0.0
        }
        
        return features

class CandidaMorphologyClassifier:
    """
    Classifier for Candida morphological types
    """
    
    def __init__(self):
        self.feature_extractor = MorphologicalFeatureExtractor()
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        self.feature_names = [
            'area', 'perimeter', 'aspect_ratio', 'solidity', 
            'extent', 'circularity', 'eccentricity', 'compactness'
        ]
        
    def classify_by_rules(self, features):
        """
        Improved rule-based classification using data-driven thresholds
        """
        aspect_ratio = features['aspect_ratio']
        area = features['area']
        solidity = features.get('solidity', 0.8)  # Default if not available
        circularity = features.get('circularity', 0.5)  # Default if not available
        
        # Improved rules based on actual data analysis (analyze_morphology.py results)
        # These thresholds are derived from analyzing 11,953 actual detections
        if aspect_ratio < 1.3 and circularity > 0.3 and area < 2000:
            return 'yeast'  # Round, smaller cells
        elif aspect_ratio >= 1.3 and aspect_ratio < 2.0 and area > 150:
            return 'pseudohyphae'  # Moderately elongated
        elif aspect_ratio >= 2.0 and area > 150:
            return 'true_hyphae'  # Highly elongated
        elif area < 100:  # Very small detections
            return 'unknown'  # Too small to classify reliably
        else:
            return 'yeast'  # Default for reasonable-sized detections
    
    def classify(self, contour):
        """
        Classify a single contour using rule-based approach
        """
        if cv2.contourArea(contour) < 20:  # Minimum area threshold
            return 'unknown'
        
        features = self.feature_extractor.extract_features(contour)
        if features is None:
            return 'unknown'
        
        return self.classify_by_rules(features)
    
    def prepare_training_data(self, image_dir, mask_dir, detection_csv):
        """
        Prepare training data with morphological features
        """
        # Load detection results
        detections_df = pd.read_csv(detection_csv)
        
        training_data = []
        
        for _, row in detections_df.iterrows():
            image_name = row['image_name']
            
            # Load mask
            mask_name = image_name.replace('.jp2', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Find contours in the bounding box region
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Extract region of interest
            roi_mask = mask[y1:y2, x1:x2]
            
            # Find contours in ROI
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use largest contour in the region
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 20:  # Minimum area threshold
                    # Extract features
                    features = self.feature_extractor.extract_features(largest_contour)
                    
                    # Rule-based classification for ground truth
                    morphology_type = self.classify_by_rules(features)
                    
                    features['morphology_type'] = morphology_type
                    features['detection_id'] = row['detection_id']
                    features['image_name'] = image_name
                    
                    training_data.append(features)
        
        return pd.DataFrame(training_data)
    
    def train_classifier(self, training_data):
        """
        Train machine learning classifier
        """
        # Filter out unknown classifications for training
        clean_data = training_data[training_data['morphology_type'] != 'unknown']
        
        if len(clean_data) < 10:
            print("Not enough clean training data for ML classification")
            return False
        
        # Prepare features and labels
        X = clean_data[self.feature_names].values
        y = clean_data['morphology_type'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        return True
    
    def classify_all_detections(self, detection_csv, mask_dir):
        """
        Classify all detected regions using both rule-based and ML approaches
        """
        detections_df = pd.read_csv(detection_csv)
        results = []
        
        for _, row in detections_df.iterrows():
            image_name = row['image_name']
            
            # Load mask
            mask_name = image_name.replace('.jp2', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Extract region of interest
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            roi_mask = mask[y1:y2, x1:x2]
            
            # Find contours
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 20:
                    # Extract features
                    features = self.feature_extractor.extract_features(largest_contour)
                    
                    # Rule-based classification
                    rule_based_class = self.classify_by_rules(features)
                    
                    # ML classification (if trained)
                    ml_class = 'unknown'
                    try:
                        feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
                        ml_class = self.classifier.predict(feature_vector)[0]
                    except:
                        pass
                    
                    result = {
                        'image_name': image_name,
                        'detection_id': row['detection_id'],
                        'xmin': row['xmin'], 'ymin': row['ymin'],
                        'xmax': row['xmax'], 'ymax': row['ymax'],
                        'rule_based_classification': rule_based_class,
                        'ml_classification': ml_class,
                        **features
                    }
                    
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath):
        """Save trained classifier"""
        joblib.dump(self.classifier, filepath)
        
    def load_model(self, filepath):
        """Load trained classifier"""
        self.classifier = joblib.load(filepath)

def visualize_morphology_distribution(classification_df, output_dir):
    """
    Create visualizations of morphological classification results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Rule-based classification distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    rule_counts = classification_df['rule_based_classification'].value_counts()
    plt.pie(rule_counts.values, labels=rule_counts.index, autopct='%1.1f%%')
    plt.title('Rule-based Classification Distribution')
    
    # Feature correlation heatmap
    plt.subplot(1, 2, 2)
    feature_cols = ['area', 'perimeter', 'aspect_ratio', 'solidity', 
                    'extent', 'circularity', 'eccentricity', 'compactness']
    corr_matrix = classification_df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Morphology type characteristics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Aspect ratio by type
    axes[0,0].boxplot([classification_df[classification_df['rule_based_classification'] == t]['aspect_ratio'].values 
                      for t in ['yeast', 'pseudohyphae', 'true_hyphae']])
    axes[0,0].set_xticklabels(['Yeast', 'Pseudohyphae', 'True Hyphae'])
    axes[0,0].set_title('Aspect Ratio by Morphology Type')
    axes[0,0].set_ylabel('Aspect Ratio')
    
    # Area by type
    axes[0,1].boxplot([classification_df[classification_df['rule_based_classification'] == t]['area'].values 
                      for t in ['yeast', 'pseudohyphae', 'true_hyphae']])
    axes[0,1].set_xticklabels(['Yeast', 'Pseudohyphae', 'True Hyphae'])
    axes[0,1].set_title('Area by Morphology Type')
    axes[0,1].set_ylabel('Area (pixels²)')
    
    # Circularity by type
    axes[1,0].boxplot([classification_df[classification_df['rule_based_classification'] == t]['circularity'].values 
                      for t in ['yeast', 'pseudohyphae', 'true_hyphae']])
    axes[1,0].set_xticklabels(['Yeast', 'Pseudohyphae', 'True Hyphae'])
    axes[1,0].set_title('Circularity by Morphology Type')
    axes[1,0].set_ylabel('Circularity')
    
    # Solidity by type
    axes[1,1].boxplot([classification_df[classification_df['rule_based_classification'] == t]['solidity'].values 
                      for t in ['yeast', 'pseudohyphae', 'true_hyphae']])
    axes[1,1].set_xticklabels(['Yeast', 'Pseudohyphae', 'True Hyphae'])
    axes[1,1].set_title('Solidity by Morphology Type')
    axes[1,1].set_ylabel('Solidity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function for morphological classification
    """
    # Paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    detection_csv = os.path.join(project_root, 'results', 'detection', 'candida_detections.csv')
    mask_dir = os.path.join(project_root, 'results', 'segmentation_masks')
    output_dir = os.path.join(project_root, 'results', 'classification_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize classifier
    classifier = CandidaMorphologyClassifier()
    
    # Classify all detections
    print("Classifying Candida morphological types...")
    classification_results = classifier.classify_all_detections(detection_csv, mask_dir)
    
    if len(classification_results) == 0:
        print("No valid detections found for classification")
        return
    
    # Save results
    results_path = os.path.join(output_dir, 'morphological_classification.csv')
    classification_results.to_csv(results_path, index=False)
    print(f"Classification results saved to: {results_path}")
    
    # Generate summary statistics
    print("\nMorphological Classification Summary:")
    print(classification_results['rule_based_classification'].value_counts())
    
    # Create visualizations
    visualize_morphology_distribution(classification_results, output_dir)
    print(f"Visualizations saved to: {output_dir}")
    
    # Train ML classifier
    print("\nPreparing training data for ML classifier...")
    training_data = classifier.prepare_training_data(
        os.path.join(project_root, 'data'), mask_dir, detection_csv
    )
    
    if len(training_data) > 0:
        success = classifier.train_classifier(training_data)
        if success:
            model_path = os.path.join(output_dir, 'morphology_classifier.pkl')
            classifier.save_model(model_path)
            print(f"Trained classifier saved to: {model_path}")

if __name__ == "__main__":
    main()