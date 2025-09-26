import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def merge_data(classification_path, quality_path):
    """
    Merges classification results with image quality scores.
    
    Args:
        classification_path (str): Path to the classification results CSV.
        quality_path (str): Path to the image quality scores CSV.
        
    Returns:
        pd.DataFrame: A merged DataFrame, or None if merging fails.
    """
    print(f"Reading classification data from: {classification_path}")
    class_df = pd.read_csv(classification_path)
    
    print(f"Reading image quality data from: {quality_path}")
    quality_df = pd.read_csv(quality_path)

    # The classification CSV has the full path, quality CSV has just the filename.
    # We need to create a common 'filename' column to merge on.
    if 'image_path' in class_df.columns and 'filename' not in class_df.columns:
        class_df['filename'] = class_df['image_path'].apply(lambda p: Path(p).name)
    
    if 'filename' not in class_df.columns or 'filename' not in quality_df.columns:
        print("Error: A common 'filename' column could not be found or created in both CSVs.")
        return None
        
    merged_df = pd.merge(class_df, quality_df, on='filename', how='left')
    
    missing_count = merged_df['sharpness'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} classification entries did not have a matching quality score.")
    else:
        print("Successfully merged classification and quality data.")
        
    return merged_df

def plot_quality_distributions(df, output_dir):
    """
    Generates and saves distribution plots for each quality metric,
    comparing correct vs. incorrect classifications.
    """
    print(f"Generating quality distribution plots in: {output_dir}")
    
    for score in ['sharpness', 'brisque', 'niqe']:
        if score not in df.columns or df[score].isna().all():
            print(f"Skipping plot for '{score}': column not found or all values are NaN.")
            continue

        plt.figure(figsize=(10, 6))
        
        correct_scores = df[df['is_correct'] == True][score].dropna()
        incorrect_scores = df[df['is_correct'] == False][score].dropna()

        if not correct_scores.empty:
            sns.kdeplot(data=correct_scores, label='Correct', color='green', fill=True, alpha=0.2)
        if not incorrect_scores.empty:
            sns.kdeplot(data=incorrect_scores, label='Incorrect', color='red', fill=True, alpha=0.2)
        
        plt.title(f'Distribution of {score.capitalize()} for Correct vs. Incorrect Predictions')
        plt.xlabel(f'{score.capitalize()} Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path = Path(output_dir) / f'{score}_distribution_by_correctness.png'
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved {save_path.name}")

def generate_summary_tables(df, output_dir):
    """
    Generates and saves summary statistic tables for each quality metric.
    """
    print(f"Generating summary statistic tables in: {output_dir}")

    for score in ['sharpness', 'brisque', 'niqe']:
        if score in df.columns and not df[score].isna().all():
            # Summary by correctness
            stats_correctness = df.groupby('is_correct')[score].describe().round(2)
            stats_correctness.index = stats_correctness.index.map({True: 'Correct', False: 'Incorrect'})
            print(f"\n--- {score.capitalize()} Statistics (by Correctness) ---")
            print(stats_correctness.to_string())
            save_path = Path(output_dir) / f'{score}_summary_by_correctness.csv'
            stats_correctness.to_csv(save_path)
            print(f"  - Saved {save_path.name}")
            
            # Summary by true label
            stats_label = df.groupby('true_label')[score].describe().round(2)
            stats_label.index = stats_label.index.map({0: 'Healthy', 1: 'Sapro'}) # Assuming 0=healthy, 1=sapro
            print(f"\n--- {score.capitalize()} Statistics (by True Label) ---")
            print(stats_label.to_string())
            save_path = Path(output_dir) / f'{score}_summary_by_label.csv'
            stats_label.to_csv(save_path)
            print(f"  - Saved {save_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the relationship between image quality and model classification performance."
    )
    parser.add_argument(
        "classification_csv",
        type=str,
        help="Path to the CSV file from 'analyse_misclassifications.py' containing prediction results."
    )
    parser.add_argument(
        "quality_csv",
        type=str,
        help="Path to the CSV file from 'image_quality_analysis.py' containing quality scores."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quality_analysis_results",
        help="Directory to save the generated plots and summary tables."
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Merge the data
    merged_df = merge_data(args.classification_csv, args.quality_csv)
    
    if merged_df is not None and not merged_df.empty:
        # Generate plots
        plot_quality_distributions(merged_df, args.output_dir)
        
        # Generate tables
        generate_summary_tables(merged_df, args.output_dir)
        
        print(f"\nAnalysis complete. All outputs saved in '{args.output_dir}'")
    else:
        print("Could not proceed with analysis due to empty or invalid data.")

if __name__ == '__main__':
    main()
