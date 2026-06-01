import numpy as np
from sklearn.metrics import f1_score


def find_best_f1_threshold(
    y_true,
    y_score,
) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)

    f1_scores = [
        f1_score(
            y_true,
            y_score >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])
