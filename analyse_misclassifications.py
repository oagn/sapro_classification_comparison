import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import argparse

# Define class names for consistency
CLASS_NAMES = ['healthy', 'sapro'] 
# Ensure this order matches the model's output and directory structure
SAPRO_INDEX = CLASS_NAMES.index('sapro')
HEALTHY_INDEX = CLASS_NAMES.index('healthy')

def load_model(model_path):
    """Loads a saved Keras model from the specified path.

    Args:
        model_path (str): Path to the saved Keras model file (.keras or .h5).

    Returns:
        keras.Model: The loaded Keras model.
    """
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

def prepare_image(image_path, target_size=(224, 224)):
    """Loads, resizes, and preprocesses a single image for model prediction.

    Preprocessing includes converting to RGB, resizing, converting to numpy array,
    normalizing pixel values to [0, 1] by dividing by 255.0, and adding a batch dimension.

    **Important**: Ensure this normalization matches the preprocessing used during
    the training of the model being analyzed.

    Args:
        image_path (str or Path): Path to the image file.
        target_size (tuple): Target size (height, width) to resize the image.

    Returns:
        np.array: The preprocessed image array ready for model prediction 
                  (shape: 1, height, width, 3).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    # Add batch dimension and normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_misclassifications(model, data_dir):
    """Analyzes model predictions across image files in specified subdirectories.

    Assumes a directory structure like:
        data_dir/
            healthy/...
            sapro/...
    
    Iterates through images, gets predictions, calculates saprolegnia probability,
    determines correctness, and compiles results into a DataFrame.

    Args:
        model (keras.Model): The loaded Keras model to use for predictions.
        data_dir (str): Path to the root directory containing 'healthy' and 'sapro' subdirs.

    Returns:
        tuple: (results_df, false_positives_df, false_negatives_df)
            results_df (pd.DataFrame): DataFrame containing results for all images.
            false_positives_df (pd.DataFrame): Subset of results_df for false positives.
            false_negatives_df (pd.DataFrame): Subset of results_df for false negatives.
    """
    print(f"Analyzing misclassifications in directory: {data_dir}")
    # Get all image paths based on global CLASS_NAMES
    sapro_paths = list(Path(data_dir).glob(f'{CLASS_NAMES[SAPRO_INDEX]}/**/*.jpg')) + \
                 list(Path(data_dir).glob(f'{CLASS_NAMES[SAPRO_INDEX]}/**/*.jpeg')) + \
                 list(Path(data_dir).glob(f'{CLASS_NAMES[SAPRO_INDEX]}/**/*.png'))
    healthy_paths = list(Path(data_dir).glob(f'{CLASS_NAMES[HEALTHY_INDEX]}/**/*.jpg')) + \
                   list(Path(data_dir).glob(f'{CLASS_NAMES[HEALTHY_INDEX]}/**/*.jpeg')) + \
                   list(Path(data_dir).glob(f'{CLASS_NAMES[HEALTHY_INDEX]}/**/*.png'))
    
    results = []
    target_img_size = tuple(model.input.shape[1:3]) # Get target size from model input
    print(f"Using target image size from model: {target_img_size}")

    # Process sapro images
    print(f"Processing {len(sapro_paths)} {CLASS_NAMES[SAPRO_INDEX]} images...")
    for path in sapro_paths:
        try:
            img_array = prepare_image(path, target_size=target_img_size)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            sapro_prob = pred[0][SAPRO_INDEX] # Use index
            
            results.append({
                'image_path': str(path),
                'true_label': SAPRO_INDEX, # Use index
                'predicted_label': pred_class,
                'sapro_probability': sapro_prob,
                'is_correct': int(pred_class == SAPRO_INDEX) # Use index
            })
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Process healthy images
    print(f"Processing {len(healthy_paths)} {CLASS_NAMES[HEALTHY_INDEX]} images...")
    for path in healthy_paths:
        try:
            img_array = prepare_image(path, target_size=target_img_size)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            sapro_prob = pred[0][SAPRO_INDEX] # Still get sapro probability
            
            results.append({
                'image_path': str(path),
                'true_label': HEALTHY_INDEX, # Use index
                'predicted_label': pred_class,
                'sapro_probability': sapro_prob,
                'is_correct': int(pred_class == HEALTHY_INDEX) # Use index
            })
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Create DataFrame and verify columns
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("Warning: No images processed successfully. Result DataFrame is empty.")
        # Return empty dataframes to avoid errors later
        return results_df, results_df.iloc[0:0], results_df.iloc[0:0] 
        
    print("\nDataFrame columns:", results_df.columns.tolist())
    print("Number of rows:", len(results_df))
    
    # Analyze results
    print("\nOverall Performance:")
    print(f"Accuracy: {(results_df['is_correct'].mean()):.3f}")
    
    # Confusion matrix analysis
    # Use indices for comparison
    false_positives = results_df[
        (results_df['true_label'] == HEALTHY_INDEX) & 
        (results_df['predicted_label'] == SAPRO_INDEX)
    ]
    false_negatives = results_df[
        (results_df['true_label'] == SAPRO_INDEX) & 
        (results_df['predicted_label'] == HEALTHY_INDEX)
    ]
    
    print(f"\nNumber of False Positives ({CLASS_NAMES[HEALTHY_INDEX]} predicted as {CLASS_NAMES[SAPRO_INDEX]}): {len(false_positives)}")
    print(f"Number of False Negatives ({CLASS_NAMES[SAPRO_INDEX]} predicted as {CLASS_NAMES[HEALTHY_INDEX]}): {len(false_negatives)}")
    
    return results_df, false_positives, false_negatives

def plot_misclassified_examples(df, category='false_positives', num_examples=5):
    """Plots example images from a DataFrame of misclassified samples.

    Sorts examples based on saprolegnia probability and displays the top N images.

    Args:
        df (pd.DataFrame): DataFrame containing misclassification results 
                           (must include 'true_label', 'predicted_label', 
                           'sapro_probability', 'image_path').
        category (str): Type of misclassification to plot ('false_positives' 
                        or 'false_negatives').
        num_examples (int): Maximum number of examples to plot.
    """
    print(f"\nPlotting {num_examples} examples for category: {category}")
    if category == 'false_positives':
        # Healthy predicted as Sapro (High sapro probability is "more wrong")
        examples = df[
            (df['true_label'] == HEALTHY_INDEX) & 
            (df['predicted_label'] == SAPRO_INDEX)
        ].sort_values('sapro_probability', ascending=False)
        title_prefix = "False Positives"
    elif category == 'false_negatives':
        # Sapro predicted as Healthy (Low sapro probability is "more wrong")
        examples = df[
            (df['true_label'] == SAPRO_INDEX) & 
            (df['predicted_label'] == HEALTHY_INDEX)
        ].sort_values('sapro_probability', ascending=True)
        title_prefix = "False Negatives"
    else:
        print(f"Error: Invalid category '{category}' for plotting.")
        return

    if examples.empty:
        print(f"No examples found for category: {category}")
        return
        
    num_to_plot = min(num_examples, len(examples))
    
    fig, axes = plt.subplots(1, num_to_plot, figsize=(4 * num_to_plot, 4))
    
    # Handle case where there's only one subplot
    if num_to_plot == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(examples.head(num_to_plot).iterrows()):
        try:
            img = Image.open(row['image_path'])
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"True: {CLASS_NAMES[int(row['true_label'])]}\nPred: {CLASS_NAMES[int(row['predicted_label'])]}\nProb(Sapro)={row['sapro_probability']:.3f}")
        except Exception as e:
            print(f"Error loading/plotting image {row['image_path']}: {e}")
            axes[i].set_title("Error loading image")
            axes[i].axis('off')
            
    plt.suptitle(f"{title_prefix} (Top {num_to_plot} examples)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def extract_features(model, image_path, target_size=(224, 224)):
    """Extracts features from a specified layer of the model for a single image.

    By default, extracts from the second-to-last layer (model.layers[-2]), 
    assuming this is the layer before the final classification layer.

    Args:
        model (keras.Model): The loaded Keras model.
        image_path (str or Path): Path to the image file.
        target_size (tuple): Target size (height, width) to resize the image 
                             (should match the size used in prepare_image).

    Returns:
        np.array: The extracted feature vector for the image.
    """
    # --- Feature Extractor Model ---
    # Assumption: The second-to-last layer contains the desired features.
    # This might need adjustment depending on the specific model architecture.
    try:
        feature_layer = model.layers[-2]
        feature_model = keras.Model(
            inputs=model.input,
            outputs=feature_layer.output 
        )
        print(f"Extracting features from layer: {feature_layer.name} (index -2)")
    except Exception as e:
        print(f"Error creating feature extraction model from layer -2: {e}")
        print("Cannot proceed with feature extraction.")
        return None
    
    # --- Prepare Image ---
    try:
        img_array = prepare_image(image_path, target_size=target_size)
    except Exception as e:
        print(f"Error preparing image {image_path} for feature extraction: {e}")
        return None
        
    # --- Extract Features ---
    try:
        features = feature_model.predict(img_array, verbose=0)
        return features[0]  # Remove batch dimension
    except Exception as e:
        print(f"Error predicting features for {image_path}: {e}")
        return None

def plot_probability_distributions(results_df):
    """Plots the distribution of predicted 'sapro' probabilities.
    
    Separates distributions for correctly and incorrectly classified samples
    for both the true 'sapro' and true 'healthy' classes.

    Args:
        results_df (pd.DataFrame): DataFrame with prediction results, requiring 
                                 'true_label', 'is_correct', 'sapro_probability'.
    """
    print("\nPlotting probability distributions...")
    plt.figure(figsize=(12, 6))
    
    # Sapro class predictions (True Label = SAPRO_INDEX)
    sapro_subset = results_df[results_df['true_label'] == SAPRO_INDEX]
    sapro_correct = sapro_subset[sapro_subset['is_correct'] == True]['sapro_probability']
    sapro_incorrect = sapro_subset[sapro_subset['is_correct'] == False]['sapro_probability']
    
    # Healthy class predictions (True Label = HEALTHY_INDEX)
    healthy_subset = results_df[results_df['true_label'] == HEALTHY_INDEX]
    healthy_correct = healthy_subset[healthy_subset['is_correct'] == True]['sapro_probability']
    healthy_incorrect = healthy_subset[healthy_subset['is_correct'] == False]['sapro_probability']
    
    # Plot distributions
    plt.subplot(1, 2, 1)
    if not sapro_correct.empty: sns.kdeplot(data=sapro_correct, label=f'Correct {CLASS_NAMES[SAPRO_INDEX]}', color='green')
    if not sapro_incorrect.empty: sns.kdeplot(data=sapro_incorrect, label=f'Incorrect {CLASS_NAMES[SAPRO_INDEX]}', color='red')
    plt.title(f'{CLASS_NAMES[SAPRO_INDEX]} Class Predictions')
    plt.xlabel(f'Probability of {CLASS_NAMES[SAPRO_INDEX]} Class')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if not healthy_correct.empty: sns.kdeplot(data=healthy_correct, label=f'Correct {CLASS_NAMES[HEALTHY_INDEX]}', color='blue')
    if not healthy_incorrect.empty: sns.kdeplot(data=healthy_incorrect, label=f'Incorrect {CLASS_NAMES[HEALTHY_INDEX]}', color='orange')
    plt.title(f'{CLASS_NAMES[HEALTHY_INDEX]} Class Predictions')
    plt.xlabel(f'Probability of {CLASS_NAMES[SAPRO_INDEX]} Class') # X-axis is still prob of sapro
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_clusters(results_df, model, n_clusters=3):
    """Performs feature extraction, PCA, and KMeans clustering on misclassified images.

    Visualizes the clusters using PCA and shows example images from each cluster.

    Args:
        results_df (pd.DataFrame): DataFrame with prediction results.
        model (keras.Model): The loaded Keras model (used for feature extraction).
        n_clusters (int): The number of clusters for KMeans.

    Returns:
        tuple: (clusters, X_pca)
            clusters (np.array): Array of cluster assignments for misclassified samples.
            X_pca (np.array): PCA-reduced coordinates (2D) for misclassified samples.
            Returns (None, None) if feature extraction fails or no misclassified samples exist.
    """
    print(f"\nPerforming clustering analysis on misclassified images (k={n_clusters})...")
    # Get misclassified images
    misclassified = results_df[results_df['is_correct'] == False].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if misclassified.empty:
        print("No misclassified images found to cluster.")
        return None, None

    target_img_size = tuple(model.input.shape[1:3]) # Get target size from model

    # Extract features for all misclassified images
    print(f"Extracting features for {len(misclassified)} misclassified images...")
    features_list = []
    valid_indices = [] # Keep track of images where feature extraction succeeded
    for idx, img_path in enumerate(misclassified['image_path']):
        features = extract_features(model, img_path, target_size=target_img_size)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
        else:
             print(f"Skipping image {img_path} due to feature extraction error.")
    
    if not features_list:
        print("Feature extraction failed for all misclassified images. Cannot perform clustering.")
        return None, None
        
    # Filter the misclassified dataframe to only include successfully processed images
    misclassified_processed = misclassified.iloc[valid_indices].reset_index(drop=True)
    
    # Convert to numpy array and normalize
    print("Scaling features...")
    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce dimensionality for visualization
    print("Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform clustering
    print("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Explicitly set n_init
    clusters = kmeans.fit_predict(X_scaled)
    misclassified_processed['cluster'] = clusters # Add cluster assignment
    
    # --- Plot Clusters ---
    print("Plotting clusters...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster ID')
    
    # Add true class information (using N/S for Healthy/Sapro)
    for i, (_, row) in enumerate(misclassified_processed.iterrows()):
        true_label_char = 'S' if row['true_label'] == SAPRO_INDEX else 'H'
        plt.annotate(
            true_label_char,
            (X_pca[i, 0], X_pca[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.9
        )
    
    plt.title(f'PCA Visualization of Misclassified Image Clusters (k={n_clusters})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    # --- Show Example Images from Each Cluster ---
    print("\nShowing example images from each cluster...")
    for cluster_id in range(n_clusters):
        cluster_subset = misclassified_processed[misclassified_processed['cluster'] == cluster_id]
        if cluster_subset.empty:
            print(f"Cluster {cluster_id} is empty.")
            continue
            
        cluster_imgs = cluster_subset['image_path'].values[:5] # Show up to 5 examples
        
        fig, axes = plt.subplots(1, len(cluster_imgs), figsize=(4 * len(cluster_imgs), 4))
        if len(cluster_imgs) == 1:
            axes = [axes]
            
        plt.suptitle(f'Example Images from Cluster {cluster_id} ({len(cluster_subset)} images total)')
        
        for ax, img_path in zip(axes, cluster_imgs):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')
            except Exception as e:
                 print(f"Error loading image {img_path} for cluster example: {e}")
                 ax.set_title("Error loading")
                 ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    return clusters, X_pca

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze misclassifications of a Keras image classification model.")
    parser.add_argument("-m", "--model_path", required=True, 
                        help="Path to the saved Keras model file (.keras or .h5).")
    parser.add_argument("-d", "--data_dir", required=True, 
                        help=f"Path to the data directory containing subdirs: '{CLASS_NAMES[HEALTHY_INDEX]}' and '{CLASS_NAMES[SAPRO_INDEX]}'.")
    parser.add_argument("-o", "--output_csv", default="misclassification_analysis.csv", 
                        help="Path to save the output CSV file with detailed results.")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, 
                        help="Number of clusters for KMeans analysis of misclassified images.")
    parser.add_argument("--num_examples", type=int, default=5, 
                        help="Number of example misclassified images to plot per category.")

    args = parser.parse_args()

    # --- Execution --- 
    print("--- Starting Misclassification Analysis ---")
    # Load model once
    model = load_model(args.model_path)
    
    # Analyze all misclassifications
    results_df, false_positives, false_negatives = analyze_misclassifications(
        model, args.data_dir
    )
    
    if not results_df.empty:
        # Plot probability distributions
        plot_probability_distributions(results_df)
        
        # Perform clustering analysis
        clusters, pca_coords = analyze_clusters(results_df, model, n_clusters=args.n_clusters)
        
        # Plot misclassified examples
        plot_misclassified_examples(results_df, category='false_positives', num_examples=args.num_examples)
        plot_misclassified_examples(results_df, category='false_negatives', num_examples=args.num_examples)
        
        # Save results
        print(f"\nSaving detailed results to: {args.output_csv}")
        results_df.to_csv(args.output_csv, index=False)
    else:
        print("\nSkipping plotting and saving as no results were generated.")

    print("--- Analysis Complete ---")
