import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Load the saved Keras model
    """
    model = keras.models.load_model(model_path)
    return model

def prepare_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    # Add batch dimension and normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_misclassifications(model, data_dir):
    """
    Analyze misclassifications across the entire dataset
    """
    # Get all image paths
    sapro_paths = list(Path(data_dir).glob('sapro/**/*.jpg')) + \
                 list(Path(data_dir).glob('sapro/**/*.jpeg')) + \
                 list(Path(data_dir).glob('sapro/**/*.png'))
    healthy_paths = list(Path(data_dir).glob('healthy/**/*.jpg')) + \
                   list(Path(data_dir).glob('healthy/**/*.jpeg')) + \
                   list(Path(data_dir).glob('healthy/**/*.png'))
    
    results = []
    
    # Process sapro images
    print(f"Processing {len(sapro_paths)} sapro images...")
    for path in sapro_paths:
        try:
            img_array = prepare_image(path)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            sapro_prob = pred[0][1]  # Assuming sapro is class 1
            
            results.append({
                'image_path': str(path),
                'true_label': 1,  # sapro
                'predicted_label': pred_class,
                'sapro_probability': sapro_prob,
                'is_correct': int(pred_class == 1)  # Explicitly cast to int
            })
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Process healthy images
    print(f"Processing {len(healthy_paths)} healthy images...")
    for path in healthy_paths:
        try:
            img_array = prepare_image(path)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            sapro_prob = pred[0][1]
            
            results.append({
                'image_path': str(path),
                'true_label': 0,  # healthy
                'predicted_label': pred_class,
                'sapro_probability': sapro_prob,
                'is_correct': int(pred_class == 0)  # Explicitly cast to int
            })
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Create DataFrame and verify columns
    results_df = pd.DataFrame(results)
    print("\nDataFrame columns:", results_df.columns.tolist())
    print("Number of rows:", len(results_df))
    
    # Analyze results
    print("\nOverall Performance:")
    print(f"Accuracy: {(results_df['is_correct'].mean()):.3f}")
    
    # Confusion matrix analysis
    false_positives = results_df[
        (results_df['true_label'] == 0) & 
        (results_df['predicted_label'] == 1)
    ]
    false_negatives = results_df[
        (results_df['true_label'] == 1) & 
        (results_df['predicted_label'] == 0)
    ]
    
    print(f"\nNumber of False Positives (Healthy predicted as sapro): {len(false_positives)}")
    print(f"Number of False Negatives (Sapro predicted as healthy): {len(false_negatives)}")
    
    return results_df, false_positives, false_negatives

def plot_misclassified_examples(df, category='false_positives', num_examples=5):
    """
    Plot misclassified examples
    category: 'false_positives' or 'false_negatives'
    """
    if category == 'false_positives':
        examples = df[
            (df['true_label'] == 0) & 
            (df['predicted_label'] == 1)
        ].sort_values('sapro_probability', ascending=False)
    else:
        examples = df[
            (df['true_label'] == 1) & 
            (df['predicted_label'] == 0)
        ].sort_values('sapro_probability')
    
    # Plot top examples
    fig, axes = plt.subplots(1, min(num_examples, len(examples)), 
                            figsize=(4*min(num_examples, len(examples)), 4))
    
    # Handle case where there's only one example
    if min(num_examples, len(examples)) == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(examples.head(num_examples).iterrows()):
        img = Image.open(row['image_path'])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Prob(sapro)={row['sapro_probability']:.3f}")
    
    plt.suptitle(f"{'False Positives' if category=='false_positives' else 'False Negatives'}")
    plt.tight_layout()
    plt.show()

def extract_features(model, image_path, target_size=(224, 224)):
    """
    Extract features from the second-to-last layer of the model
    """
    # Create a new model that outputs features from the layer before classification
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # Assuming last layer is the classification layer
    )
    
    # Prepare image
    img_array = prepare_image(image_path)
    
    # Extract features
    features = feature_model.predict(img_array, verbose=0)
    return features[0]  # Remove batch dimension

def plot_probability_distributions(results_df):
    """
    Plot probability distributions for correct and incorrect predictions
    """
    plt.figure(figsize=(12, 6))
    
    # Sapro class predictions
    sapro_correct = results_df[
        (results_df['true_label'] == 1) & 
        (results_df['is_correct'] == True)
    ]['sapro_probability']
    sapro_incorrect = results_df[
        (results_df['true_label'] == 1) & 
        (results_df['is_correct'] == False)
    ]['sapro_probability']
    
    # Non-sapro class predictions
    nonsapro_correct = results_df[
        (results_df['true_label'] == 0) & 
        (results_df['is_correct'] == True)
    ]['sapro_probability']
    nonsapro_incorrect = results_df[
        (results_df['true_label'] == 0) & 
        (results_df['is_correct'] == False)
    ]['sapro_probability']
    
    # Plot distributions
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=sapro_correct, label='Correct Sapro', color='green')
    sns.kdeplot(data=sapro_incorrect, label='Incorrect Sapro', color='red')
    plt.title('Sapro Class Predictions')
    plt.xlabel('Probability of Sapro Class')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=nonsapro_correct, label='Correct Non-sapro', color='green')
    sns.kdeplot(data=nonsapro_incorrect, label='Incorrect Non-sapro', color='red')
    plt.title('Non-sapro Class Predictions')
    plt.xlabel('Probability of Sapro Class')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_clusters(results_df, model, n_clusters=3):
    """
    Perform clustering analysis on misclassified images
    """
    from sklearn.cluster import KMeans
    
    # Get misclassified images
    misclassified = results_df[results_df['is_correct'] == False]
    
    # Extract features for all misclassified images
    features_list = []
    for img_path in misclassified['image_path']:
        features = extract_features(model, img_path)
        features_list.append(features)
    
    # Convert to numpy array and normalize
    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by cluster
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    
    # Add true class information
    for i, (_, row) in enumerate(misclassified.iterrows()):
        plt.annotate(
            'S' if row['true_label'] == 1 else 'N',
            (X_pca[i, 0], X_pca[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            alpha=0.7
        )
    
    plt.title('Clusters of Misclassified Images')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
    # Show example images from each cluster
    for cluster in range(n_clusters):
        cluster_imgs = misclassified[clusters == cluster]['image_path'].values[:5]
        
        fig, axes = plt.subplots(1, len(cluster_imgs), 
                                figsize=(4*len(cluster_imgs), 4))
        if len(cluster_imgs) == 1:
            axes = [axes]
            
        plt.suptitle(f'Example Images from Cluster {cluster}')
        
        for ax, img_path in zip(axes, cluster_imgs):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return clusters, X_pca

if __name__ == "__main__":
    MODEL_PATH = "path/to/your/best/model"
    DATA_DIR = "path/to/image/directory"
    
    # Load model once
    model = load_model(MODEL_PATH)
    
    # Analyze all misclassifications
    results_df, false_positives, false_negatives = analyze_misclassifications(
        model, DATA_DIR
    )
    
    # Plot probability distributions
    plot_probability_distributions(results_df)
    
    # Perform clustering analysis
    clusters, pca_coords = analyze_clusters(results_df, model, n_clusters=3)
    
    # Original visualizations
    plot_misclassified_examples(results_df, category='false_positives', num_examples=5)
    plot_misclassified_examples(results_df, category='false_negatives', num_examples=5)
    
    # Save results
    results_df.to_csv('misclassification_analysis.csv', index=False)
