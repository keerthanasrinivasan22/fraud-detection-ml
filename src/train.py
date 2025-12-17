from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import (
    load_data,
    split_data,
    compute_pr_auc,
    find_best_threshold
)


DATA_PATH = "dataset/creditcard.csv"
TARGET_PRECISION = 0.85


def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = compute_pr_auc(y_test, y_proba)
    threshold, precision, recall = find_best_threshold(
        y_test, y_proba, TARGET_PRECISION
    )

    return pr_auc, threshold, precision, recall


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = compute_pr_auc(y_test, y_proba)
    threshold, precision, recall = find_best_threshold(
        y_test, y_proba, TARGET_PRECISION
    )

    return pr_auc, threshold, precision, recall


if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Logistic Regression...")
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    print("LR PR-AUC:", lr_results[0])
    print("LR threshold:", lr_results[1])
    print("LR precision:", lr_results[2])
    print("LR recall:", lr_results[3])

    print("\nTraining Random Forest...")
    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    print("RF PR-AUC:", rf_results[0])
    print("RF threshold:", rf_results[1])
    print("RF precision:", rf_results[2])
    print("RF recall:", rf_results[3])
