from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision_illicit": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall_illicit": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_illicit": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
        "threshold": threshold,
    }
