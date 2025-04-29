# Saprolegnia Classification Comparison

## Overview

This project trains and evaluates several deep learning models (using Keras with a JAX backend) for the binary classification of fish images into 'healthy' and 'sapro' (infected with <i>Saprolegnia</i> spp.). It utilizes k-fold cross-validation with optional stratified grouping and oversampling techniques. The project also includes a separate script for detailed analysis of model misclassifications.

## Project Structure

```
.
├── analyse_misclassifications.py # Script to analyze model errors, plot results, and perform clustering.
├── config.yaml                   # Configuration file for training parameters, paths, and model settings.
├── data_loader.py                # Handles data loading, preprocessing, augmentation, and cross-validation splitting.
├── evaluate.py                   # Calculates and saves evaluation metrics (accuracy, F1, confusion matrix, etc.).
├── models.py                     # Defines model architectures (base models + classification head).
├── sapro.sh                      # SLURM batch script to run the training pipeline on an HPC cluster.
├── sapro_classification.py       # Main script for running the cross-validation training and evaluation pipeline.
├── train.py                      # Contains functions for training models (frozen and unfrozen phases).
└── README.md                     # This file.
```

## Setup

### 1. Environment

This project uses a Conda environment. The required environment is assumed to be named `keras-jax` as specified in `sapro.sh`.

```bash
# Activate the environment
conda activate keras-jax
```

Key dependencies include:
*   Python
*   Keras 3 (configured with JAX backend)
*   JAX
*   TensorFlow (for tf.data pipeline and image loading)
*   keras-cv (for RandAugment and FocalLoss)
*   scikit-learn (for metrics, splitting, PCA, StandardScaler, KMeans)
*   pandas
*   numpy
*   PyYAML
*   Matplotlib
*   Seaborn
*   Pillow (PIL)
*   imblearn (if using SMOTE was intended, otherwise can be removed from env)

### 2. Data Preparation

*   **Training Data:** The training script (`sapro_classification.py` via `sapro.sh`) expects image data organized in class subdirectories within the path specified by `data.train_dir` in `config.yaml`. Example:
    ```
    /path/to/train_data/
    ├── healthy/
    │   ├── img1.jpg
    │   └── img2.png
    └── sapro/
        ├── img3.jpeg
        └── img4.jpg
    ```
*   **Metadata:** A CSV file containing metadata (including user/group IDs and stratification columns) is required, specified by `data.metadata_path` in `config.yaml`. It needs a column matching image filenames (e.g., `data_row.external_id`).
*   **Analysis Data:** The analysis script (`analyse_misclassifications.py`) expects a similar structure in the directory provided via its `-d` argument, corresponding to the classes defined (e.g., `healthy/` and `sapro/`).

## Configuration (`config.yaml`)

This file controls the entire training process. Key sections to configure:

*   `data`:
    *   `output_dir`: Where all results (logs, model checkpoints, summaries) will be saved. **Ensure this path exists or is writable.**
    *   `train_dir`: Path to the training image directory.
    *   `metadata_path`: Path to the metadata CSV file.
    *   `weights_path`: Set to `null` to use default Imagenet weights, or provide a path to `.h5` / `.keras` file for custom pre-trained weights.
    *   `batch_size`, `augmentation_magnitude`, `class_names`.
    *   `group_column`, `stratify_columns`: Columns in the metadata CSV used for `StratifiedGroupKFold` or `StratifiedKFold`.
*   `models`: Define parameters for each model architecture to be tested (`img_size`, `num_dense_layers`, `unfreeze_layers`).
*   `training`:
    *   Epoch counts (`initial_epochs`, `fine_tuning_epochs`).
    *   Learning rates (`learning_rate`, `fine_tuning_lr`).
    *   `focal_loss_gamma`.
    *   `early_stopping_patience`.
    *   Cross-validation settings (`use_groups`, `n_folds`).
    *   Oversampling settings (`use_oversampling`, `sampling_strategy`, `threshold_ratio`).

## Running the Training Pipeline

The primary way to run the training is using the SLURM batch script on a compatible HPC environment.

1.  **Modify `sapro.sh` (if necessary):** Adjust SBATCH directives (time, memory, GPU count, account), the Conda environment name (`keras-jax`), or the `input_dir` if your code resides elsewhere relative to `$HOME`.
2.  **Submit the job:**
    ```bash
    sbatch sapro.sh
    ```
3.  **Output:** The script will create a working directory in `/scratch/$USER/`, copy the necessary files, run the `sapro_classification.py` script, and save all outputs to the `output_dir` specified in `config.yaml`. This includes:
    *   A summary text file (`cv_results_summary_*.txt`).
    *   Confusion matrix plots and classification reports for each fold in subdirectories (`fold_1/`, `fold_2/`, etc.).
    *   The best model checkpoint (`.keras` file) for each fold's fine-tuning phase.

## Analyzing Misclassifications

After training, you can analyze the performance of a specific saved model using `analyse_misclassifications.py`.

1.  **Activate the environment:** `conda activate keras-jax`
2.  **Run the script:**

    ```bash
    python analyse_misclassifications.py \
        -m /path/to/your/best_model_fold_X_unfrozen.keras \
        -d /path/to/your/analysis_data_directory/ \
        -o analysis_results.csv \
        -k 4 \
        --num_examples 10
    ```

    **Arguments:**
    *   `-m`, `--model_path` ( **Required**): Path to the saved Keras model file you want to analyze.
    *   `-d`, `--data_dir` ( **Required**): Path to the directory containing 'healthy' and 'sapro' subdirectories for analysis.
    *   `-o`, `--output_csv` (Optional): Path to save the detailed CSV results (default: `misclassification_analysis.csv`).
    *   `-k`, `--n_clusters` (Optional): Number of clusters for KMeans analysis (default: 3).
    *   `--num_examples` (Optional): Number of example misclassified images to plot (default: 5).

3.  **Output:**
    *   Plots (displayed interactively): Probability distributions, PCA cluster visualization, example misclassified images per category, example images per cluster.
    *   A CSV file (specified by `-o`) containing detailed prediction results for every image analyzed. 