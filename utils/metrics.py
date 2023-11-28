# utils/metrics.py

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    return accuracy, f1, precision, recall, roc_auc
