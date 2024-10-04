import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def evaluate_model(model, test_ds):
    y_pred = []
    y_true = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))

    # Print classification report (includes precision, recall, f1-score for each class)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Calculate and print F1 score for each class
    f1_scores = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(f1_scores):
        print(f"F1 score for class {i}: {f1:.4f}")

    # Calculate and print macro and weighted F1 scores
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Macro F1 score: {macro_f1:.4f}")
    print(f"Weighted F1 score: {weighted_f1:.4f}")

    return {
        'f1_scores': f1_scores,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }