# computer-vision-fish-disease

## Overview

This project trains and evaluates deep learning models for the binary classification of fish images into 'healthy' and 'sapro' (infected with *Saprolegnia* spp.). It includes scripts for training, evaluation, and detailed analysis of model performance in relation to image quality.

## Project Structure

The project has been streamlined to keep all executable scripts in a single directory.

```
.
├── notebooks/
│   └── download_dataset.ipynb
├── scripts/
│   ├── analyse_classification_quality.py
│   ├── analyse_misclassifications.py
│   ├── image_quality_analysis.py
│   ├── train.py
│   ├── evaluate.py
│   └── ... (other scripts)
├── hpc/
│   └── sapro.sh
├── config.yaml                 # Configuration file for training parameters.
├── environment.yml             # Conda environment file for all tasks.
├── CITATION.cff                # Citation file for the repository.
└── README.md                   # This file.
```

## Setup

### 1. Environment

This project uses a single Conda environment to manage all dependencies for both machine learning and image quality analysis tasks.

-   `sapro-env`: Used for all scripts. Based on Python 3.10.

**Create the environment:**
Run the following command from your project's root directory:
```bash
# Create the Conda environment
conda env create -f environment.yml
```

### 2. Data Preparation

To reproduce the results of this study, you must first download the publicly available image dataset and generate the taxonomic subsets.

1.  **Run the Download Notebook:** Open and run the `notebooks/download_dataset.ipynb` notebook. This notebook contains all the necessary steps to download the dataset and create taxonomic subsets using symlinks.
2.  **Configure Paths:** Once the data is downloaded, update your `config.yaml` to point to the correct directory for the analysis you wish to run.

## Running the Training Pipeline

There are two primary ways to run the training pipeline: locally for testing, or on an HPC cluster for full-scale training.

### Running Locally

1.  **Activate the environment:**
    ```bash
    conda activate sapro-env
    ```
2.  **Run the training script:**
    ```bash
    python scripts/sapro_classification.py
    ```

### Running on an HPC Cluster (using SLURM)

1.  **Modify `hpc/sapro.sh` (if necessary):** Adjust SBATCH directives (time, memory, GPU count), the Conda environment name (`sapro-env`), or other paths as needed.
2.  **Submit the job:**
    ```bash
    sbatch hpc/sapro.sh
    ```

## Analyzing Model Performance and Image Quality

This project provides a reproducible workflow for analyzing how image quality metrics correlate with model performance.

The process involves three main steps, all run in the **`sapro-env`** environment.

1.  **Generate Prediction Results:**
    First, run the `analyse_misclassifications.py` script. This will produce a detailed CSV file of the model's predictions on your dataset.
    ```bash
    conda activate sapro-env
    python scripts/analyse_misclassifications.py \
        -m /path/to/your/best_model.keras \
        -d /path/to/your/analysis_data_directory/ \
        -o prediction_results.csv
    ```

2.  **Generate Image Quality Scores:**
    Next, run the `image_quality_analysis.py` script on the *same* data directory. This will produce a CSV containing sharpness, BRISQUE, and NIQE scores for every image.
    ```bash
    conda activate sapro-env
    python scripts/image_quality_analysis.py \
        /path/to/your/analysis_data_directory/ \
        --output_csv quality_scores.csv \
        --brisque_model data/brisque_model_live.yml \
        --brisque_range data/brisque_range_live.yml \
        --niqe_params data/niqe_image_params.mat
    ```

3.  **Run the Combined Analysis:**
    Finally, use the `analyse_classification_quality.py` script to combine these two results.
    ```bash
    conda activate sapro-env
    python scripts/analyse_classification_quality.py \
        prediction_results.csv \
        quality_scores.csv \
        --output_dir quality_analysis_results
    ```

    **Output:** This will create a new directory (e.g., `quality_analysis_results/`) containing:
    *   Box plots (`.png`) for each quality metric, comparing correct vs. incorrect predictions.
    *   Summary tables (`.csv`) detailing the statistics for each quality metric, grouped by correctness and true label. 
