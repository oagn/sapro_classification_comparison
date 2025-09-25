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
├── download_dataset.ipynb        # Jupyter Notebook to download the public dataset from Zenodo.
├── image_quality_analysis.py     # Script to calculate image sharpness scores.
├── requirements.txt              # A list of required Python packages for the project.
├── CITATION.cff                  # Citation file for the repository.
├── taxonomic_info.csv            # Taxonomic data used by the download notebook for creating subsets.
└── README.md                     # This file.
```

## Setup

### 1. Environment

This project uses a Conda environment. The required environment is assumed to be named `keras-jax` as specified in `sapro.sh`.

1.  **Create and activate the environment:**
    ```bash
    # If you have an environment.yml file:
    # conda env create -f environment.yml
    conda activate keras-jax
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Data Preparation

To reproduce the results of this study, you must first download the publicly available image dataset and generate the taxonomic subsets.

1.  **Run the Download Notebook:** Open and run the `download_dataset.ipynb` notebook. This notebook contains all the necessary steps to:
    a.  Guide you to download the metadata from our Zenodo repository.
    b.  Use the included `taxonomic_info.csv` file to assign taxonomic data to each image.
    c.  Download all the publicly available images into a base directory (`downloaded_dataset/`).
    d.  Automatically create all taxonomic subsets (e.g., `subsets/E_salmo_geq_10_in_both/`) using symlinks to save disk space.

2.  **Configure Paths:** Once the data is downloaded, update your `config.yaml` to point to the correct directory for the analysis you wish to run. For example:
    ```yaml
    data:
      train_dir: 'subsets/E_salmo_geq_10_in_both/' # Path to a specific subset for training
      metadata_path: 'path/to/your/metadata.csv' # Metadata for grouping/stratification
    ```

#### Note on Extending the Dataset

The data for this project was collated by searching public APIs (Flickr, iNaturalist, GBIF) for relevant keywords and taxonomic names. The resulting image metadata was then manually labeled for the presence of visible signs of *Saprolegnia spp.* infection using Labelbox. If you wish to extend this work with more recent images, you would follow a similar process of programmatic searching, data collation, and manual labeling.

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

## Additional Analyses

### Analyzing Misclassifications

After training, you can analyze the performance of a specific saved model using `analyse_misclassifications.py`. This script can also integrate image sharpness data to explore its impact on model performance.

1.  **Activate the environment:** `conda activate keras-jax`
2.  **(Optional) Generate Sharpness Scores:** First, run the `image_quality_analysis.py` script on your image directory to generate a CSV of sharpness scores.
    ```bash
    python image_quality_analysis.py /path/to/your/image_folder --output_csv sharpness_scores.csv
    ```
3.  **Run the analysis script:**

    ```bash
    python analyse_misclassifications.py \
        -m /path/to/your/best_model.keras \
        -d /path/to/your/analysis_data_directory/ \
        -o analysis_results.csv \
        -s sharpness_scores.csv # Optional: include sharpness data
    ```

    **Arguments:**
    *   `-m`, `--model_path` ( **Required**): Path to the saved Keras model file you want to analyze.
    *   `-d`, `--data_dir` ( **Required**): Path to the directory containing 'healthy' and 'sapro' subdirectories for analysis.
    *   `-o`, `--output_csv` (Optional): Path to save the detailed CSV results.
    *   `-k`, `--n_clusters` (Optional): Number of clusters for KMeans analysis.
    *   `--num_examples` (Optional): Number of example misclassified images to plot.
    *   `-s`, `--sharpness_csv` (Optional): Path to the CSV file with image sharpness scores.

4.  **Output:**
    *   Plots (displayed interactively): Probability distributions, sharpness distributions, PCA cluster visualization, example misclassified images.
    *   A CSV file (specified by `-o`) containing detailed prediction results. 