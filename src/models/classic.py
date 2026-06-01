from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import SEED


def build_dummy_classifier():
    return DummyClassifier(
        strategy="prior",
    )


def build_logistic_regression():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
