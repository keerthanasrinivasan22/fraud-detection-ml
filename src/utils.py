import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score


def load_data(path):
    """
    Load credit card fraud dataset.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def compute_pr_auc(y_true, y_proba):
    """
    Compute PR-AUC score.
    """
    return average_precision_score(y_true, y_proba)


def find_best_threshold(y_true, y_proba, target_precision):
    """
    Find threshold that maximizes recall
    while maintaining target precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    valid_idxs = np.where(precision[:-1] >= target_precision)[0]

    if len(valid_idxs) == 0:
        return 1.0, precision[-1], recall[-1]

    best_idx = valid_idxs[np.argmax(recall[valid_idxs])]

    return thresholds[best_idx], precision[best_idx], recall[best_idx]
