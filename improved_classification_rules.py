
def classify_by_rules_improved(self, features):
    """
    Improved rule-based classification using data-driven thresholds
    """
    aspect_ratio = features['aspect_ratio']
    area = features['area']
    solidity = features.get('solidity', 0.8)  # Default if not available
    circularity = features.get('circularity', 0.5)  # Default if not available
    
    # Improved rules based on actual data analysis
    if aspect_ratio < 1.2 and circularity > 0.4 and area < 1756:
        return 'yeast'  # Round, smaller cells
    elif aspect_ratio >= 1.2 and aspect_ratio < 1.4 and area > 210:
        return 'pseudohyphae'  # Moderately elongated
    elif aspect_ratio >= 1.4 and area > 210:
        return 'true_hyphae'  # Highly elongated
    else:
        return 'unknown'  # Unclear morphology
