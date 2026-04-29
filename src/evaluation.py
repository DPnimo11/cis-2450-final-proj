import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_classifier(model, X_test, y_test, label: str = "Model") -> dict[str, object]:
    """Generate the core metrics used in the current baseline notebook."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "label": label,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "classification_report": report,
        "classification_report_df": pd.DataFrame(report).transpose(),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
