import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

def evaluate_model(model, test_path, class_names, batch_size=32, img_size=224, output_path=None):
    y_pred = []
    y_true = []

    class_map = {name: idx for idx, name in enumerate(class_names)}
    inv_class = {v: k for k, v in class_map.items()}
    class_ids = sorted(inv_class.values())

    for i, spp_class in enumerate(class_names):
        print(f"\nEvaluating class '{spp_class}' ({(i+1)}/ {len(class_names)})")
        img_generator = tf.keras.preprocessing.image_dataset_from_directory(
            test_path + '/' + spp_class, 
            labels=None,
            label_mode=None,
            batch_size=int(batch_size), 
            image_size=(int(img_size), int(img_size)),
            shuffle=False)
        preds = model.predict(img_generator)
        y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
        y_pred.extend(y_pred_tmp)
        y_true.extend([spp_class] * len(y_pred_tmp))

    # calculate metrics
    print("\n\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    # plot confusion matrix
    confusion_matrix = confusion_matrix(y_true, y_pred, normalize = "true")
    rcParams.update({'figure.autolayout': True})
    nc = len(class_ids)
    if nc > 20: # adjust font size with many classes
        font_size = 7 if nc < 35 else 5
        rcParams.update({'font.size': font_size})
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_ids)
    cm_display.plot(cmap=plt.cm.Blues, include_values = len(class_ids) < 8, values_format = '.2g') # only include values with few classes
    plt.xticks(rotation=90, ha='center')
    plt.savefig(output_path + "/confusion_matrix.png")

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