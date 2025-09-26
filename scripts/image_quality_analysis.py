import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
from brisque import BRISQUE
from skimage import io
from skimage.metrics import mean_squared_error
from skimage.restoration import estimate_sigma

# In this newer environment, niqe should be available.
try:
    from skimage.metrics import natural_image_quality_evaluator as niqe
except ImportError:
    # This is now less likely, but good practice to keep the fallback.
    print("Warning: NIQE could not be imported. NIQE scores will be NaN.")
    def niqe(image): return np.nan

def calculate_sharpness(image_gray):
    """Calculates sharpness using the variance of the Laplacian."""
    if image_gray is None: return 0.0
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()

def calculate_niqe(image_gray):
    """Calculates the NIQE score."""
    if image_gray is None: return np.nan
    # NIQE function from skimage expects values between 0-255
    return niqe(image_gray)

def calculate_noise(image_gray):
    """Estimates the noise level in a grayscale image."""
    if image_gray is None: return np.nan
    # estimate_sigma uses the wavelet method, robust to image content
    return estimate_sigma(image_gray, channel_axis=None)

def analyze_image_quality(data_dir, output_csv):
    """Analyzes multiple quality metrics for all images in a directory."""
    dir_path = Path(data_dir)
    image_paths = sorted(list(dir_path.glob('**/*.[jJ][pP][gG]')) + 
                         list(dir_path.glob('**/*.[jJ][pP][eE][gG]')) + 
                         list(dir_path.glob('**/*.[pP][nN][gG]')))

    if not image_paths:
        print(f"No images found in directory: {data_dir}")
        return

    print(f"Found {len(image_paths)} images to analyze for Sharpness, BRISQUE, NIQE, Resolution, and Noise.")
    
    # Instantiate the BRISQUE model once
    brisque_scorer = BRISQUE(url=False)
    
    results = []

    for path in tqdm(image_paths, desc="Analyzing Image Quality"):
        try:
            # Read image once in both formats
            image_gray = io.imread(path, as_gray=True)
            image_rgb = io.imread(path) # imquality lib expects RGB

            # Get image dimensions
            height, width = image_rgb.shape[:2]

            # Convert grayscale from [0,1] float to [0,255] uint8 for NIQE and sharpness
            if image_gray.dtype in (np.float64, np.float32):
                 image_gray = (image_gray * 255).astype(np.uint8)

            sharpness = calculate_sharpness(image_gray)
            brisque_score = brisque_scorer.score(image_rgb)
            niqe_score = calculate_niqe(image_gray)
            noise = calculate_noise(image_gray)
            
            results.append({
                'filename': path.name,
                'sharpness': sharpness,
                'brisque': brisque_score,
                'niqe': niqe_score,
                'width': width,
                'height': height,
                'noise': noise
            })
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            results.append({
                'filename': path.name,
                'sharpness': np.nan,
                'brisque': np.nan,
                'niqe': np.nan,
                'width': np.nan,
                'height': np.nan,
                'noise': np.nan
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nImage quality analysis complete. Results saved to {output_csv}")
    print("\nSummary statistics:")
    print(df.describe())

def main():
    parser = argparse.ArgumentParser(description="Analyze image quality metrics (Sharpness, BRISQUE, NIQE, Resolution, Noise).")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the directory containing the images to analyze."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="image_quality_scores.csv",
        help="Path to the output CSV file. (default: image_quality_scores.csv)"
    )
    args = parser.parse_args()
    analyze_image_quality(args.data_dir, args.output_csv)

if __name__ == '__main__':
    main()
