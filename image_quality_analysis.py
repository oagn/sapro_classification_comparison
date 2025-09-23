import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm

def calculate_sharpness(image_path):
    """ Calculates the sharpness of an image using the variance of the Laplacian.

    Args:
        image_path (str or Path): The path to the image file.

    Returns:
        float: The sharpness score. A higher value means a sharper image.
               Returns 0.0 if the image cannot be read. """
    try:
        # Read the image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return 0.0
            
        # Compute the Laplacian of the image and then return the variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0.0

def analyze_directory_sharpness(data_dir, output_csv):
    """ Analyzes the sharpness of all images in a directory and saves the results to a CSV.

    Args:
        data_dir (str): Path to the root directory containing images.
        output_csv (str): Path to save the output CSV file. """
    dir_ = Path(data_dir)
    
    # Recursively find all common image files
    image_paths = list(dir_.glob(r'**/*.jpg'))
    image_paths.extend(list(dir_.glob(r'**/*.JPG')))
    image_paths.extend(list(dir_.glob(r'**/*.jpeg')))
    image_paths.extend(list(dir_.glob(r'**/*.png')))
    image_paths.extend(list(dir_.glob(r'**/*.PNG')))

    if not image_paths:
        print(f"No images found in directory: {data_dir}")
        return

    print(f"Found {len(image_paths)} images to analyze.")

    results = []
    
    # Use tqdm for a progress bar
    for path in tqdm(image_paths, desc="Analyzing image sharpness"):
        sharpness_score = calculate_sharpness(path)
        results.append({
            'file_path': str(path),
            'sharpness_score': sharpness_score
        })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSharpness analysis complete. Results saved to {output_csv}")
    print("\nSummary statistics of sharpness scores:")
    print(df['sharpness_score'].describe())

def main():
    """ Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze image sharpness in a directory using the variance of the Laplacian method."
    )
    parser.add_argument(
        "data_dir", 
        type=str, 
        help="Path to the directory containing the images to analyze."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="sharpness_scores.csv", 
        help="Path to the output CSV file. (default: sharpness_scores.csv)"
    )
    
    args = parser.parse_args()
    
    analyze_directory_sharpness(args.data_dir, args.output_csv)

if __name__ == '__main__':
    main()
