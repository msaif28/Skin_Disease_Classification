# utils/metrics.py

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob):
    # Ensure that y_true, y_pred, and y_prob are numpy arrays and of appropriate types
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob)  # Convert list to numpy array if not already

    # Print and check the shape and max value of y_true and shape of y_prob for debugging
    num_classes = len(np.unique(y_true))  # Determine the number of unique classes
    print(f"Unique classes: {num_classes}, y_true max: {np.max(y_true)}, y_prob shape: {y_prob.shape}")

    # Ensure the shape of y_prob is correct
    if y_prob.shape[1] != num_classes:
        raise ValueError(f"Number of classes in probabilities does not match number of unique classes. Expected {num_classes}, got {y_prob.shape[1]}")

    # Check and handle the binarization of y_true
    if num_classes > 2:  # Binarize for multiclass
        y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
    else:
        y_true_binarized = y_true  # For binary, it's the same as y_true
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr')
    return accuracy, f1, precision, recall, roc_auc