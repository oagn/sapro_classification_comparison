import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.stats import mannwhitneyu

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

def plot_quality_boxplots(df, output_dir):
    """
    Generates and saves box plots for each quality metric, comparing 
    correct vs. incorrect classifications within each true class.
    """
    print(f"Generating quality box plots in: {output_dir}")
    
    # Map numeric labels to readable strings for plotting
    df['Classification'] = df['is_correct'].map({1: 'Correct', 0: 'Incorrect'})
    df['True Label'] = df['true_label'].map({0: 'Healthy', 1: 'Sapro'})
    
    for score in ['sharpness', 'brisque', 'niqe', 'width', 'height']:
        if score not in df.columns or df[score].isna().all():
            print(f"Skipping plot for '{score}': column not found or all values are NaN.")
            continue

        plt.figure(figsize=(12, 8))
        
        # Custom blue palette: light for 'Correct', darker for 'Incorrect'
        custom_palette = {"Correct": "#a1c9f4", "Incorrect": "#225ea8"}
        
        # Use hue to create faceted plots for each class
        sns.boxplot(x='True Label', y=score, hue='Classification', data=df, 
                    order=['Healthy', 'Sapro'], hue_order=['Correct', 'Incorrect'], 
                    palette=custom_palette)
        
        # Increase font size for labels and ticks
        plt.xlabel('True Label', fontsize=14)
        plt.ylabel(f'{score.capitalize()} Score', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Update filename to reflect the more detailed plot
        save_path = Path(output_dir) / f'{score}_boxplot_by_class_and_correctness.png'
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved {save_path.name}")


def generate_summary_tables(df, output_dir):
    """
    Generates and saves a detailed summary statistic table for each quality metric,
    grouped by both true label and classification correctness.
    """
    print(f"Generating detailed summary statistic tables in: {output_dir}")

    for score in ['sharpness', 'brisque', 'niqe', 'width', 'height']:
        if score in df.columns and not df[score].isna().all():
            # Create a combined summary table grouped by both class and correctness
            stats_combined = df.groupby(['True Label', 'Classification'])[score].describe().round(2)
            
            # Ensure a consistent order for readability
            if not stats_combined.empty:
                stats_combined = stats_combined.reindex(
                    pd.MultiIndex.from_product(
                        [['Healthy', 'Sapro'], ['Correct', 'Incorrect']], 
                        names=['True Label', 'Classification']
                    )
                )

            print(f"\n--- {score.capitalize()} Statistics (by Class and Correctness) ---")
            print(stats_combined.to_string())
            
            # Update filename to reflect the new, more detailed grouping
            save_path = Path(output_dir) / f'{score}_summary_by_class_and_correctness.csv'
            stats_combined.to_csv(save_path)
            print(f"  - Saved {save_path.name}")


def perform_stratified_mann_whitney_u_tests(df, output_dir):
    """
    Performs STRATIFIED Mann-Whitney U tests to compare quality scores between
    correct and incorrect classifications for EACH true class separately.
    """
    print("\nPerforming STRATIFIED Mann-Whitney U tests (Correct vs Incorrect within each class)...")

    # These columns should be created by plot_quality_boxplots, but we can ensure they exist
    if 'Classification' not in df.columns:
        df['Classification'] = df['is_correct'].map({1: 'Correct', 0: 'Incorrect'})
    if 'True Label' not in df.columns:
        df['True Label'] = df['true_label'].map({0: 'Healthy', 1: 'Sapro'})

    results = []
    quality_metrics = ['sharpness', 'brisque', 'niqe', 'width', 'height']

    for score in quality_metrics:
        if score not in df.columns or df[score].isna().all():
            print(f"Skipping Mann-Whitney U test for '{score}': column not found or all values are NaN.")
            continue

        for label in ['Healthy', 'Sapro']:
            class_df = df[df['True Label'] == label]
            
            correct_scores = class_df[class_df['Classification'] == 'Correct'][score].dropna()
            incorrect_scores = class_df[class_df['Classification'] == 'Incorrect'][score].dropna()

            if len(correct_scores) > 0 and len(incorrect_scores) > 0:
                try:
                    stat, p_value = mannwhitneyu(correct_scores, incorrect_scores, alternative='two-sided')
                    results.append({
                        'Metric': score,
                        'True Label': label,
                        'U-statistic': stat,
                        'p-value': p_value
                    })
                except ValueError as e:
                    print(f"Could not perform test for {score} in {label}: {e}")
            else:
                print(f"Skipping test for {score} in {label} due to insufficient data.")

    if not results:
        print("No stratified Mann-Whitney U tests could be performed.")
        return

    results_df = pd.DataFrame(results)
    
    print("\n--- Stratified Mann-Whitney U Test Results ---")
    print(results_df.to_string())

    save_path = Path(output_dir) / 'mann_whitney_u_test_results_stratified.csv'
    results_df.to_csv(save_path, index=False)
    print(f"  - Saved {save_path.name}")

def perform_pooled_mann_whitney_u_tests(df, output_dir):
    """
    Performs POOLED Mann-Whitney U tests to compare quality scores between
    ALL correct and ALL incorrect classifications, regardless of true class.
    """
    print("\nPerforming POOLED Mann-Whitney U tests (ALL Correct vs ALL Incorrect)...")

    # Ensure 'Classification' column exists
    if 'Classification' not in df.columns:
        df['Classification'] = df['is_correct'].map({1: 'Correct', 0: 'Incorrect'})

    results = []
    quality_metrics = ['sharpness', 'brisque', 'niqe', 'width', 'height']

    for score in quality_metrics:
        if score not in df.columns or df[score].isna().all():
            print(f"Skipping pooled test for '{score}': column not found or all values are NaN.")
            continue

        # Get scores for ALL correct and incorrect images
        correct_scores = df[df['Classification'] == 'Correct'][score].dropna()
        incorrect_scores = df[df['Classification'] == 'Incorrect'][score].dropna()

        if len(correct_scores) > 0 and len(incorrect_scores) > 0:
            try:
                stat, p_value = mannwhitneyu(correct_scores, incorrect_scores, alternative='two-sided')
                results.append({
                    'Metric': score,
                    'U-statistic': stat,
                    'p-value': p_value
                })
            except ValueError as e:
                print(f"Could not perform pooled test for {score}: {e}")
        else:
            print(f"Skipping pooled test for {score} due to insufficient data.")

    if not results:
        print("No pooled Mann-Whitney U tests could be performed.")
        return

    results_df = pd.DataFrame(results)
    
    print("\n--- Pooled Mann-Whitney U Test Results ---")
    print(results_df.to_string())

    save_path = Path(output_dir) / 'mann_whitney_u_test_results_pooled.csv'
    results_df.to_csv(save_path, index=False)
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
        plot_quality_boxplots(merged_df, args.output_dir)
        
        # Generate tables
        generate_summary_tables(merged_df, args.output_dir)
        
        # Perform statistical tests (both types)
        perform_stratified_mann_whitney_u_tests(merged_df, args.output_dir)
        perform_pooled_mann_whitney_u_tests(merged_df, args.output_dir)
        
        print(f"\nAnalysis complete. All outputs saved in '{args.output_dir}'")
    else:
        print("Could not proceed with analysis due to empty or invalid data.")

if __name__ == '__main__':
    main()