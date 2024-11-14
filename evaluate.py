import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

def evaluate_model(model, test_dataset, class_names, batch_size=None, img_size=None, output_path=None):
    """
    Evaluate model performance on test dataset
    
    Args:
        model: Trained model
        test_dataset: Either a path to test data or a TensorFlow dataset
        class_names: List of class names
        batch_size: Batch size (only used if test_dataset is a path)
        img_size: Image size (only used if test_dataset is a path)
        output_path: Path to save evaluation results
    """
    os.makedirs(output_path, exist_ok=True)
    class_ids = list(range(len(class_names)))
    all_preds = []
    all_true = []
    
    if isinstance(test_dataset, str):
        # If test_dataset is a path, process each class directory
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
        # If test_dataset is already a TensorFlow dataset
        print("\nEvaluating test dataset...")
        for images, labels in test_dataset:
            preds = model.predict(images)
            y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
            y_true_tmp = [label.numpy().argmax() for label in labels]
            
            all_preds.extend(y_pred_tmp)
            all_true.extend(y_true_tmp)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_preds)
    conf_matrix = confusion_matrix(all_true, all_preds)
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
        'classification_report': classification_rep
    }