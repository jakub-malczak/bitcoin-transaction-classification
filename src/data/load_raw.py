from pathlib import Path

import pandas as pd

from src.config import (
    CLASSES_PATH,
    EDGES_PATH,
    FEATURES_PATH,
    FEATURE_COLUMNS,
)


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    features_df = pd.read_csv(
        path,
        header=None,
        names=FEATURE_COLUMNS,
    )

    return features_df


def load_classes(path: Path = CLASSES_PATH) -> pd.DataFrame:
    classes_df = pd.read_csv(path).rename(
        columns={
            "txId": "tx_id",
            "class": "label",
        }
    )

    classes_df["label"] = classes_df["label"].astype(str)

    return classes_df[["tx_id", "label"]]


def load_edges(path: Path = EDGES_PATH) -> pd.DataFrame:
    edges_df = pd.read_csv(path).rename(
        columns={
            "txId1": "source_tx_id",
            "txId2": "target_tx_id",
        }
    )

    return edges_df[["source_tx_id", "target_tx_id"]]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_df = load_features()
    classes_df = load_classes()
    edges_df = load_edges()

    return features_df, classes_df, edges_df