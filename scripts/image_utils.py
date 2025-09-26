"""
Robust image loading utilities for handling various formats including JP2
"""

import os
import cv2
import numpy as np
from PIL import Image
import warnings

def load_image_robust(image_path):
    """
    Robust image loading function that tries multiple methods for JP2 files.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array: Loaded image in BGR format, or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: File does not exist: {image_path}")
        return None
    
    # Method 1: Try OpenCV first (works for most formats)
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img
    except Exception as e:
        pass
    
    # Method 2: Try PIL/Pillow for JP2 files
    try:
        with Image.open(image_path) as pil_img:
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL to OpenCV format (RGB to BGR)
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
    except Exception as e:
        pass
    
    # Method 3: Try with imagecodecs (if available)
    try:
        import imagecodecs
        with open(image_path, 'rb') as f:
            data = f.read()
        
        img_array = imagecodecs.jpeg2k_decode(data)
        if img_array is not None:
            # Ensure it's 3-channel and in BGR format
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Method 4: Try with skimage
    try:
        from skimage import io
        img_array = io.imread(image_path)
        if img_array is not None:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        pass
    
    print(f"Warning: Could not load image using any method: {image_path}")
    return None

def get_image_paths_robust(root_dir):
    """
    Lists and verifies all image files in the specified directory with robust loading.
    Only returns paths of images that can actually be loaded.
    """
    image_paths = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jp2')
    failed_files = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_formats):
                full_path = os.path.join(dirpath, filename)
                
                # Test if we can load the image
                test_img = load_image_robust(full_path)
                if test_img is not None:
                    image_paths.append(full_path)
                else:
                    failed_files.append(full_path)
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files could not be loaded:")
        for failed_file in failed_files[:10]:  # Show only first 10
            print(f"  - {os.path.basename(failed_file)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more files")
    
    print(f"Successfully verified {len(image_paths)} images out of {len(image_paths) + len(failed_files)} total files.")
    return image_paths

def test_image_formats():
    """Test which image formats can be loaded successfully."""
    print("Testing image format support...")
    
    # Test OpenCV JP2 support
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Check build info for JP2 support
        build_info = cv2.getBuildInformation()
        if "JPEG 2000" in build_info or "OpenJPEG" in build_info:
            print("✓ OpenCV has JP2/JPEG2000 support")
        else:
            print("⚠ OpenCV may have limited JP2 support")
    except:
        print("✗ OpenCV not available")
    
    # Test PIL JP2 support
    try:
        from PIL import Image
        print(f"✓ PIL/Pillow available")
        
        # Check for JP2 plugin
        if 'JPEG2000' in Image.EXTENSION:
            print("✓ PIL has JPEG2000 support")
        else:
            print("⚠ PIL may have limited JP2 support")
    except:
        print("✗ PIL not available")
    
    # Test imagecodecs
    try:
        import imagecodecs
        print(f"✓ imagecodecs available")
        if hasattr(imagecodecs, 'jpeg2k_decode'):
            print("✓ imagecodecs has JPEG2000 support")
    except:
        print("⚠ imagecodecs not available")
    
    # Test skimage
    try:
        import skimage
        print(f"✓ scikit-image available")
    except:
        print("⚠ scikit-image not available")

if __name__ == "__main__":
    test_image_formats()