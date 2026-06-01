from typing import Literal

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_MODEL_FEATURE_COLUMNS,
    LOCAL_MODEL_FEATURE_COLUMNS,
)

FeatureSet = Literal["local", "all"]

LABEL_MAPPING = {
    "1": 1,  # illicit
    "2": 0,  # licit
}


def merge_features_and_labels(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
) -> pd.DataFrame:
    classes_df = classes_df.copy()

    classes_df["target"] = (
        classes_df["label"]
        .astype(str)
        .map(LABEL_MAPPING)
        .astype("Int64")
    )

    transactions_df = features_df.merge(
        classes_df,
        on="tx_id",
        how="left",
        validate="one_to_one",
    )

    return transactions_df


def get_feature_columns(feature_set: FeatureSet) -> list[str] | None:
    if feature_set == "local":
        return LOCAL_MODEL_FEATURE_COLUMNS.copy()

    if feature_set == "all":
        return ALL_MODEL_FEATURE_COLUMNS.copy()

    return None


def get_labeled_transactions(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    return transactions_df[
        transactions_df["target"].notna()
    ].copy()


def extract_xy(
    transactions_df: pd.DataFrame,
    feature_set: FeatureSet,
):
    feature_columns = get_feature_columns(feature_set)

    x = transactions_df[feature_columns]
    y = transactions_df["target"].astype(int)

    return x, y


def standardize_splits(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
):
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    return (
        scaler,
        x_train_scaled,
        x_validation_scaled,
        x_test_scaled,
    )
