"""
Configuration — all paths, hyperparameters, and thresholds for the pipeline.
"""
import os
from pathlib import Path

# ── Versioning ────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "v6.2"

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
# Isolate outputs per pipeline version to prevent overwriting known-good submissions
OUTPUT_DIR = PROJECT_ROOT / "SUBMISSION" / "output" / PIPELINE_VERSION
DIAGNOSTICS_DIR = PROJECT_ROOT / "notebooks" / "diagnostics" / PIPELINE_VERSION
VALIDATE_SCRIPT = (
    PROJECT_ROOT / "data" / "6ab10eb3b23ba_student_resource"
    / "student_resource" / "utils" / "validate_submission.py"
)
THRESHOLD_PATH = OUTPUT_DIR / "model_threshold.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

# ── Development ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SAMPLE_FRAC = 1.0          # Lower this for sampled development runs
VAL_FRAC = 0.0             # 80/20 train/val split on S1 entities

# ── Blocking ───────────────────────────────────────────────────────────────────
BLOCKING_TOP_K = 25
TFIDF_NGRAM_RANGE = (1, 1)
TFIDF_MAX_FEATURES = 100_000

# ── Model (LightGBM) ──────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "n_estimators": 500,
    "scale_pos_weight": 1.0,   # Auto-set during training
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Threshold ──────────────────────────────────────────────────────────────────
THRESHOLD_MIN = 0.3
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

# ── Validation ─────────────────────────────────────────────────────────────────
CV_FOLDS = 5
TEST_S1_COUNT = 1_732_544
