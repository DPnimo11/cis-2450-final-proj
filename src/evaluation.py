import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def evaluate_classifier(model, X, y, label: str = "Model") -> dict[str, object]:
    y_pred = model.predict(X)
    y_proba = predict_probabilities(model, X)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return {
        "label": label,
        "predictions": y_pred,
        "probabilities": y_proba,
        "classification_report": report,
        "classification_report_df": pd.DataFrame(report).transpose(),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
    }


def make_result_row(
    *,
    scope: str,
    model_name: str,
    resampling: str,
    split_name: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "scope": scope,
        "model": model_name,
        "resampling": resampling,
        "split": split_name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }
