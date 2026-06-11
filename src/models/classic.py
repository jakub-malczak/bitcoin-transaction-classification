from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import SEED


def build_dummy_classifier():
    return DummyClassifier(
        strategy="prior",
    )


def build_logistic_regression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    class_weight=None,
    max_iter=1000,
    l1_ratio=None
):
    return LogisticRegression(
        penalty=penalty,
        C=C,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        class_weight=class_weight
    )


def build_random_forest(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        max_features=max_features,
        n_jobs=-1,
        random_state=SEED
    )


def build_xgboost(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    min_child_weight=1,
    scale_pos_weight=None,
    reg_alpha=0,
    reg_lambda=0,
):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )
