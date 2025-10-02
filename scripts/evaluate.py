import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, test_dataset, class_names, batch_size=None, img_size=None, output_path=None):
    """
    Evaluates a trained model on a given dataset and calculates common metrics.

    Can handle input as either a directory path string or a tf.data.Dataset.
    Calculates accuracy, confusion matrix, precision, recall, F1-score (per-class,
    macro, weighted), and generates a classification report.
    Optionally saves the confusion matrix plot and classification report to disk.

    Args:
        model (keras.Model): The trained Keras model to evaluate.
        test_dataset (str | tf.data.Dataset): EITHER the path to the test dataset 
            directory (structured with subdirs per class) OR a pre-processed 
            tf.data.Dataset object.
        class_names (list): A list of strings representing the class names in the 
            correct order.
        batch_size (int, optional): Batch size to use ONLY if test_dataset is a 
            directory path. Defaults to None. Should be provided if using path input.
        img_size (tuple, optional): Target image size (height, width) ONLY if 
            test_dataset is a directory path. Defaults to None. Should be 
            provided if using path input.
        output_path (str, optional): Directory path to save the confusion matrix 
            plot ('confusion_matrix.png') and classification report 
            ('classification_report.txt'). If None, files are not saved. 
            Defaults to None.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics:
            - 'accuracy': Overall accuracy.
            - 'confusion_matrix': The confusion matrix (numpy array).
            - 'classification_report': The classification report string.
            - 'precision_per_class': Precision for each class.
            - 'recall_per_class': Recall for each class.
            - 'f1_per_class': F1-score for each class.
            - 'support_per_class': Support (number of samples) for each class.
            - 'macro_f1': Macro-averaged F1-score.
            - 'weighted_f1': Weighted-averaged F1-score.
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
    class_ids = list(range(len(class_names)))
    all_preds = []
    all_true = []
    
    # --- Get Predictions and True Labels ---
    if isinstance(test_dataset, str):
        # Input is a directory path
        print(f"Evaluating model using directory: {test_dataset}")
        if not batch_size or not img_size:
            raise ValueError("batch_size and img_size must be provided when test_dataset is a directory path.")
            
        for i, spp_class in enumerate(class_names):
            print(f"\nProcessing class '{spp_class}' ({(i+1)}/ {len(class_names)}) for ground truth")
            img_generator = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(test_dataset, spp_class),
                labels=None,
                label_mode=None,
                batch_size=int(batch_size),
                image_size=(int(img_size), int(img_size)),
                shuffle=False
            )
            preds = model.predict(img_generator)
            y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
            y_true_tmp = [i] * len(y_pred_tmp)
            
            all_preds.extend(y_pred_tmp)
            all_true.extend(y_true_tmp)
    else:
        # Input is a tf.data.Dataset
        print("\nEvaluating model using provided tf.data.Dataset...")
        predictions = model.predict(test_dataset)
        
        # Collect all true labels
        true_labels = []
        for _, labels in test_dataset:
            if isinstance(labels, tuple):  # If dataset includes sample weights
                labels = labels[0]
            true_labels.extend(labels.numpy())
        
        # Convert predictions and true labels to class indices
        all_preds = [pred.argmax() for pred in predictions]
        all_true = [label.argmax() for label in true_labels]
    
    # --- Calculate Metrics ---
    print("\nCalculating evaluation metrics...")
    accuracy = accuracy_score(all_true, all_preds)
    conf_matrix = confusion_matrix(all_true, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    weighted_f1 = f1_score(all_true, all_preds, average='weighted')
    classification_rep = classification_report(all_true, all_preds, target_names=class_names)
    print(classification_rep)
    
    # --- Save Results (Optional) ---
    if output_path:
        print(f"\nSaving evaluation results to: {output_path}")
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
        plt.close()
        
        with open(os.path.join(output_path, 'classification_report.txt'), 'w') as f:
            f.write(classification_rep)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_rep,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }