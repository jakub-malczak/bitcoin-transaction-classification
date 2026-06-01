from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

FEATURES_PATH = DATASET_DIR / "elliptic_txs_features.csv"
CLASSES_PATH = DATASET_DIR / "elliptic_txs_classes.csv"
EDGES_PATH = DATASET_DIR / "elliptic_txs_edgelist.csv"

SEED = 42

TRAIN_END = 34
VALIDATION_END = 41
TEST_END = 49

ANON_FEATURE_COLUMNS = [
    f"feature_{index}"
    for index in range(1, 166)
]

FEATURE_COLUMNS = [
    "tx_id",
    "time_step",
    *ANON_FEATURE_COLUMNS,
]

ALL_MODEL_FEATURE_COLUMNS = [
    "time_step",
    *ANON_FEATURE_COLUMNS,
]

LOCAL_MODEL_FEATURE_COLUMNS = ALL_MODEL_FEATURE_COLUMNS[:94]

