from dataclasses import dataclass

import pandas as pd

from src.config import (
    TEST_END,
    TRAIN_END,
    VALIDATION_END,
)


@dataclass
class TemporalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_temporal_masks(
    transactions_df: pd.DataFrame,
    labeled_only: bool = True,
) -> dict[str, pd.Series]:
    time_step = transactions_df["time_step"]

    if labeled_only:
        allowed_rows = transactions_df["target"].notna()
    else:
        allowed_rows = pd.Series(
            True,
            index=transactions_df.index,
        )

    masks = {
        "train": (
            (time_step <= TRAIN_END)
            & allowed_rows
        ),
        "validation": (
            (time_step > TRAIN_END)
            & (time_step <= VALIDATION_END)
            & allowed_rows
        ),
        "test": (
            (time_step > VALIDATION_END)
            & (time_step <= TEST_END)
            & allowed_rows
        ),
    }

    return masks


def split_labeled_transactions(
    transactions_df: pd.DataFrame,
) -> TemporalSplits:
    masks = create_temporal_masks(
        transactions_df,
        labeled_only=True,
    )

    splits = TemporalSplits(
        train=transactions_df.loc[masks["train"]].copy(),
        validation=transactions_df.loc[masks["validation"]].copy(),
        test=transactions_df.loc[masks["test"]].copy(),
    )

    return splits