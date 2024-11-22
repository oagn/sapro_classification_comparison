import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix as sk_confusion_matrix,
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
    Evaluate model performance on test dataset or validation dataset
    """
    os.makedirs(output_path, exist_ok=True)
    class_ids = list(range(len(class_names)))
    all_preds = []
    all_true = []
    
    if isinstance(test_dataset, str):
        # Process directory path
        for i, spp_class in enumerate(class_names):
            print(f"\nEvaluating class '{spp_class}' ({(i+1)}/ {len(class_names)})")
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
        # Process TensorFlow dataset
        print("\nEvaluating validation dataset...")
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
    
    # Calculate all metrics
    accuracy = accuracy_score(all_true, all_preds)
    conf_matrix = sk_confusion_matrix(all_true, all_preds, normalize="true")
    precision, recall, f1, support = precision_recall_fscore_support(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    weighted_f1 = f1_score(all_true, all_preds, average='weighted')
    classification_rep = classification_report(all_true, all_preds, target_names=class_names)
    
    # Save results
    if output_path:
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