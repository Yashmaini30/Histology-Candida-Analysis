import os
import cv2
import yaml
import imagecodecs
import numpy as np

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config file not found at {config_full_path}")
    with open(config_full_path, 'r') as file:
        return yaml.safe_load(file)

def convert_jp2_to_png(source_dir, dest_dir):
    """
    Converts .jp2 files to .png format in a new directory structure.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    jp2_files_converted = 0
    jp2_files_failed = 0

    for dirpath, _, filenames in os.walk(source_dir):
        # Create corresponding sub-directories in the destination folder
        relative_path = os.path.relpath(dirpath, source_dir)
        destination_subdir = os.path.join(dest_dir, relative_path)
        if not os.path.exists(destination_subdir):
            os.makedirs(destination_subdir)

        for filename in filenames:
            if filename.lower().endswith('.jp2'):
                jp2_path = os.path.join(dirpath, filename)
                png_filename = os.path.splitext(filename)[0] + '.png'
                png_path = os.path.join(destination_subdir, png_filename)

                try:
                    # Read the JP2 file using imagecodecs
                    img_np = imagecodecs.imread(jp2_path)
                    
                    # Ensure it's in a valid format before writing
                    if img_np is not None and len(img_np.shape) >= 2:
                        cv2.imwrite(png_path, img_np)
                        print(f"Converted {jp2_path} -> {png_path}")
                        jp2_files_converted += 1
                    else:
                        print(f"Warning: Corrupt or empty file, skipping conversion: {jp2_path}")
                        jp2_files_failed += 1
                except Exception as e:
                    print(f"Error converting {jp2_path}: {e}")
                    jp2_files_failed += 1
    
    print(f"\nConversion summary:")
    print(f"Successfully converted: {jp2_files_converted} files.")
    print(f"Failed to convert: {jp2_files_failed} files.")

if __name__ == "__main__":
    config = load_config()
    source_data_dir = config['paths']['data_root']
    converted_data_dir = "converted_data/" # New directory for converted files
    
    print("Starting file format conversion...")
    convert_jp2_to_png(source_data_dir, converted_data_dir)
    print("Conversion complete.")