"""
V4 Pure-Ensemble Test Inference — Generate versioned V4 test outputs.

Reads validated V4 model + thresholds from cache_v4/, runs test inference
with per-country checkpointing, and writes outputs to SUBMISSION/output/v4/.

Does NOT use model_cache_1.pkl or any V3 artifacts.

Usage:
    python src/v4_inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import joblib

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SAMPLE_FRAC, OUTPUT_DIR, TEST_DIR, TEST_S1_COUNT,
    VALIDATE_SCRIPT,
)
from preprocessing.load import load_test_data
from preprocessing.clean import preprocess_dataframe
from preprocessing.transliterate import apply_transliteration
from blocking.blocker import generate_candidates
from features.similarity import extract_features
from postprocessing.threshold import format_and_save_output
from models.matcher import EntityMatcher
import subprocess

# ── Paths ────────────────────────────────────────────────────────────────────
SHARED_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'
)
V4_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache_v4'
)
V4_OUTPUT_DIR = os.path.join(str(OUTPUT_DIR), 'v4')
os.makedirs(V4_OUTPUT_DIR, exist_ok=True)
os.makedirs(V4_CACHE, exist_ok=True)

PROGRESS_PATH = os.path.join(V4_CACHE, 'test_progress_v4.txt')


def _preprocess_all(data: dict, splits: list[str]) -> dict:
    """Apply cleaning and transliteration to all dataframes in the data dict."""
    for split in splits:
        for key_suffix in ['s1', 's2', 's3']:
            key = f"{split}_{key_suffix}"
            if key in data:
                print(f"\n[preprocess] Cleaning {key} ({len(data[key]):,} records) ...")
                data[key] = preprocess_dataframe(data[key])
                is_s1 = key_suffix == 's1'
                print(f"[preprocess] Transliterating {key} (is_s1={is_s1}) ...")
                data[key] = apply_transliteration(data[key], is_s1=is_s1)
    return data


def main():
    total_start = time.time()
    timing = {}

    # ── 1. Load V4 model cache ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 INFERENCE: Loading V4 model cache")
    print("=" * 60)

    model_cache_v4_path = os.path.join(V4_CACHE, 'model_cache_v4.pkl')
    if not os.path.exists(model_cache_v4_path):
        print("[FATAL] model_cache_v4.pkl not found. Run v4_validate.py first.")
        sys.exit(1)

    v4_data = joblib.load(model_cache_v4_path)
    matcher = v4_data['matcher']
    best_threshold = v4_data['best_threshold']
    best_f05 = v4_data['best_f05']

    print(f"[v4] V4 validation F0.5: {best_f05:.6f}")
    if isinstance(best_threshold, dict):
        for c, t in sorted(best_threshold.items()):
            print(f"  Threshold ({c}): {t:.4f}")
    else:
        print(f"  Threshold: {best_threshold:.4f}")

    del v4_data
    gc.collect()

    # ── 2. Load/generate test data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 INFERENCE: Loading test data")
    print("=" * 60)

    t0 = time.time()
    test_data_path = os.path.join(SHARED_CACHE, 'test_data.pkl')
    if os.path.exists(test_data_path):
        print("[cache] Loading preprocessed test data from shared cache...")
        test_data = joblib.load(test_data_path)
    else:
        print("[load] Loading and preprocessing test data...")
        test_data = load_test_data()
        test_data = _preprocess_all(test_data, ['test'])
        joblib.dump(test_data, test_data_path)
    timing['Load test data'] = time.time() - t0

    countries = sorted(test_data['test_s1']['country'].unique())
    print(f"[v4] Test countries: {countries}")
    for c in countries:
        n = (test_data['test_s1']['country'] == c).sum()
        print(f"  {c}: {n:,} S1 entities")

    # ── 3. Check progress (resume support) ───────────────────────────────
    completed_countries = set()
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            completed_countries = set(f.read().splitlines())
        if completed_countries:
            print(f"[v4] Resuming: already completed {completed_countries}")

    tsv_path = os.path.join(V4_OUTPUT_DIR, 'matching_results.tsv')

    # ── 4. Per-country inference loop ────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 INFERENCE: Per-country test prediction")
    print("=" * 60)

    for country in countries:
        if country in completed_countries:
            print(f"\n[v4] Skipping {country} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"V4 INFERENCE: Processing {country}")
        print(f"{'='*60}")

        t0_country = time.time()

        # Block
        print(f"\n[blocking] Generating candidates for {country}...")
        country_pairs = generate_candidates(
            test_data['test_s1'], test_data['test_s2'], test_data['test_s3'],
            target_country=country,
            cache_prefix='test'
        )
        print(f"[blocking] {country}: {len(country_pairs):,} candidate pairs")

        # Features
        print(f"\n[features] Extracting features for {country}...")
        X_test = extract_features(
            country_pairs,
            test_data['test_s1'], test_data['test_s2'], test_data['test_s3']
        )
        print(f"[features] {country}: {X_test.shape[0]:,} x {X_test.shape[1]} features")

        # Predict
        print(f"\n[predict] Generating ensemble probabilities for {country}...")
        test_probs = matcher.predict_proba(X_test)
        print(f"[predict] {country}: min={test_probs.min():.6f} max={test_probs.max():.6f} mean={test_probs.mean():.6f}")

        # Format and save
        test_s1_ids = test_data['test_s1'][
            test_data['test_s1']['country'] == country
        ]['entity_id']
        append_mode = os.path.exists(tsv_path) and len(completed_countries) > 0

        preds_df = format_and_save_output(
            country_pairs, test_probs, best_threshold, test_s1_ids,
            output_dir=V4_OUTPUT_DIR,
            append_mode=append_mode,
        )

        country_time = time.time() - t0_country
        timing[f'Inference ({country})'] = country_time
        print(f"\n[v4] {country} complete in {country_time / 60:.1f} min")

        # Checkpoint
        completed_countries.add(country)
        with open(PROGRESS_PATH, 'a') as f:
            f.write(country + '\n')

        del country_pairs, X_test, test_probs, preds_df
        gc.collect()

    # ── 5. Validate row count ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 INFERENCE: Validating output shape")
    print("=" * 60)

    df_check = pd.read_csv(tsv_path, sep='\t')
    actual_rows = len(df_check)
    print(f"[v4] matching_results.tsv rows: {actual_rows:,} (expected: {TEST_S1_COUNT:,})")

    if actual_rows != TEST_S1_COUNT:
        print(f"[WARNING] Row count mismatch! {actual_rows} != {TEST_S1_COUNT}")
    else:
        print("[v4] Row count OK")

    assert list(df_check.columns) == ['source1_entity_id', 'matched_entity_ids'], \
        f"Column mismatch: {list(df_check.columns)}"
    print("[v4] Column names OK")

    n_matched = df_check['matched_entity_ids'].notna() & (df_check['matched_entity_ids'] != '')
    print(f"[v4] Matched: {n_matched.sum():,}, Singletons: {(~n_matched).sum():,}")
    del df_check
    gc.collect()

    # ── 6. Run official validator ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 INFERENCE: Running official validator")
    print("=" * 60)

    matching_path = os.path.join(V4_OUTPUT_DIR, 'matching_results.tsv')
    candidate_path = os.path.join(V4_OUTPUT_DIR, 'candidate_pairs.tsv')
    test_dir = str(TEST_DIR)

    cmd = [
        sys.executable, str(VALIDATE_SCRIPT),
        '--matching', matching_path,
        '--candidate', candidate_path,
        '--test-dir', test_dir,
    ]

    print(f"[validate] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("[validate] VALIDATION FAILED")
    else:
        print("[validate] VALIDATION PASSED")

    # ── 7. Summary ───────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("V4 INFERENCE TIMING SUMMARY:")
    for name, dur in timing.items():
        print(f"  {name:.<45} {dur/60:>6.1f} min")
    print(f"{'='*60}")
    print(f"TOTAL: {total_time/60:.1f} min")
    print(f"V4 Val F0.5: {best_f05:.6f}")
    if isinstance(best_threshold, dict):
        t_str = str({k: f"{v:.3f}" for k, v in best_threshold.items()})
    else:
        t_str = f"{best_threshold:.3f}"
    print(f"Thresholds: {t_str}")
    print(f"Validator: {'PASS' if result.returncode == 0 else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
