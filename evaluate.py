import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_ds):
    y_pred = []
    y_true = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))