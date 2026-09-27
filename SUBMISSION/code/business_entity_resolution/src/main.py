"""
Main entry point — chains all pipeline stages.

Usage:
    python src/main.py --mode train     # Train + predict on test + output files
    python src/main.py --mode predict   # Load model + predict on test
    python src/main.py --mode cv        # Cross-validation on training data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
import time
import gc
import numpy as np
import pandas as pd
import joblib

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PIPELINE_VERSION,
    SAMPLE_FRAC, OUTPUT_DIR, DIAGNOSTICS_DIR, VALIDATE_SCRIPT,
    TEST_DIR, TEST_S1_COUNT, THRESHOLD_PATH,
)
from preprocessing.load import load_train_data, load_test_data
from preprocessing.clean import preprocess_dataframe
from preprocessing.transliterate import apply_transliteration
from blocking.blocker import generate_candidates, compute_blocking_recall
from features.similarity import extract_features, generate_labels, FEATURE_NAMES
from models.matcher import EntityMatcher
# from models.reranker import TwoStageReranker
from postprocessing.threshold import sweep_threshold, format_and_save_output
from evaluation.metrics import compute_diagnostics, f05_macro


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


def run_train(skip_inference=False, loco_country=None):
    """
    Full training pipeline:
    1. Load train data with train/val split
    2. Clean + transliterate
    3. Block: generate candidate pairs
    4. Extract features + labels
    5. Train LightGBM
    6. Sweep threshold on val set
    7. Load test data, clean + transliterate
    8. Block test, extract features, predict
    9. Format and write output
    10. Run validation
    11. Generate diagnostics
    """
    total_start = time.time()
    timing_dict = {}
    sample_key = f"{SAMPLE_FRAC:g}"
    if loco_country:
        sample_key += f"_loco_{loco_country}"

    # ── 1. Load & Preprocess Data (Cached) ───────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 & 2: Loading & Preprocessing")
    print("="*60)
    
    t0_stage1 = time.time()
    train_data_path = os.path.join(CACHE_DIR, f'train_data_{sample_key}.pkl')
    if os.path.exists(train_data_path):
        print("[cache] Loading preprocessed train/val data from cache...")
        data = joblib.load(train_data_path)
    else:
        data = load_train_data(sample_frac=SAMPLE_FRAC, loco_country=loco_country)
        data = _preprocess_all(data, ['train', 'val'])
        print("[cache] Saving preprocessed train/val data to cache...")
        joblib.dump(data, train_data_path)
    timing_dict['Stage 1 & 2: Preprocessing'] = time.time() - t0_stage1

    # ── 3. Blocking ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: Blocking (TF-IDF candidate generation)")
    print("="*60)
    
    t0_stage3 = time.time()
    train_pairs_path = os.path.join(CACHE_DIR, f'train_pairs_{PIPELINE_VERSION}_{sample_key}.pkl')
    val_pairs_path = os.path.join(CACHE_DIR, f'val_pairs_{PIPELINE_VERSION}_{sample_key}.pkl')

    if os.path.exists(train_pairs_path):
        print("\n[cache] Loading TRAIN candidates from cache...")
        train_pairs, train_blocking_recall = joblib.load(train_pairs_path)
    else:
        print("\n[blocking] Generating TRAIN candidates ...")
        train_pairs = generate_candidates(data['train_s1'], data['train_s2'], data['train_s3'], cache_prefix=f'train_{PIPELINE_VERSION}_{sample_key}')
        train_blocking_recall = compute_blocking_recall(train_pairs, data['train_gt'])
        joblib.dump((train_pairs, train_blocking_recall), train_pairs_path)

    if os.path.exists(val_pairs_path):
        print("\n[cache] Loading VAL candidates from cache...")
        val_pairs, val_blocking_recall = joblib.load(val_pairs_path)
    else:
        print("\n[blocking] Generating VAL candidates ...")
        val_pairs = generate_candidates(data['val_s1'], data['val_s2'], data['val_s3'], cache_prefix=f'val_{PIPELINE_VERSION}_{sample_key}')
        val_blocking_recall = compute_blocking_recall(val_pairs, data['val_gt'])
        joblib.dump((val_pairs, val_blocking_recall), val_pairs_path)
    timing_dict['Stage 3: Blocking'] = time.time() - t0_stage3

    # ── 4. Feature extraction ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 4: Feature extraction (Cached)")
    print("="*60)
    
    t0_stage4 = time.time()
    train_feat_path = os.path.join(CACHE_DIR, f'train_feat_{PIPELINE_VERSION}_{sample_key}.pkl')
    val_feat_path = os.path.join(CACHE_DIR, f'val_feat_{PIPELINE_VERSION}_{sample_key}.pkl')

    if os.path.exists(train_feat_path):
        print("\n[cache] Loading TRAIN features from cache...")
        X_train, y_train = joblib.load(train_feat_path, mmap_mode='r')
    else:
        print("\n[features] Extracting TRAIN features ...")
        X_train = extract_features(train_pairs, data['train_s1'], data['train_s2'], data['train_s3'])
        y_train = generate_labels(train_pairs, data['train_gt'])
        joblib.dump((X_train, y_train), train_feat_path)

    if os.path.exists(val_feat_path):
        print("\n[cache] Loading VAL features from cache...")
        X_val, y_val = joblib.load(val_feat_path, mmap_mode='r')
    else:
        print("\n[features] Extracting VAL features ...")
        X_val = extract_features(val_pairs, data['val_s1'], data['val_s2'], data['val_s3'])
        y_val = generate_labels(val_pairs, data['val_gt'])
        joblib.dump((X_val, y_val), val_feat_path)
    timing_dict['Stage 4: Feature Extraction'] = time.time() - t0_stage4

    model_cache_path = os.path.join(CACHE_DIR, f'model_cache_{PIPELINE_VERSION}_{sample_key}.pkl')
    matcher = EntityMatcher()

    if os.path.exists(model_cache_path):
        print("\n[cache] Loading trained model and threshold from cache...")
        t0_stage5 = time.time()
        model_data = joblib.load(model_cache_path)
        matcher = model_data['matcher']
        best_threshold = model_data['best_threshold']
        best_f05 = model_data['best_f05']
        val_probs = model_data['val_probs']
        val_s1_ids = model_data['val_s1_ids']
        timing_dict['Stage 5 & 6: Training & Tuning (Cached)'] = time.time() - t0_stage5
    else:
        # ── 5. Train model ────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("STAGE 5: Training LightGBM")
        print("="*60)
        
        t0_stage5 = time.time()
        # Free massive memory blocks before LightGBM copies the data
        print("\n[memory] Freeing 7GB of cached DataFrames to make room for LightGBM internal datasets...")
        if 'train_pairs' in locals():
            del train_pairs
        if 'data' in locals():
            del data
        import gc; gc.collect()

        if len(X_val) == 0:
            matcher.train(X_train, y_train, None, None, feature_names=FEATURE_NAMES)
        else:
            matcher.train(X_train, y_train, X_val, y_val, feature_names=FEATURE_NAMES)
        timing_dict['Stage 5: Training LightGBM'] = time.time() - t0_stage5

        # ── 6. Threshold tuning on val set ────────────────────────────────────
        print("\n" + "="*60)
        print("STAGE 6: Threshold tuning")
        print("="*60)
        
        t0_stage6 = time.time()
        if len(val_pairs) == 0:
            print("\n[threshold] 100% Run Detected (val_frac=0.0): Skipping validation sweep.")
            print("[threshold] Loading pre-tuned 80/20 thresholds...")
            import json
            with open(THRESHOLD_PATH, "r") as f:
                try:
                    best_threshold = json.load(f)
                except:
                    best_threshold = float(f.read().strip())
            best_f05 = 0.0
            val_probs = np.array([])
            val_s1_ids = set()
            
            print("\n[cache] Saving trained model and threshold to cache...")
            joblib.dump({
                "matcher": matcher,
                "best_threshold": best_threshold,
                "best_f05": best_f05,
                "val_probs": val_probs,
                "val_s1_ids": val_s1_ids
            }, model_cache_path)
        else:
            val_probs = matcher.predict_proba(X_val)
        
            # Reload data if we deleted it to save memory for LightGBM
            if 'data' not in locals():
                print("\n[memory] Reloading data from cache for reranker...")
                data = joblib.load(train_data_path)

            # (Reranker was removed in V4 - bypassing straight to Threshold)
        
            val_s1_ids = set(data['val_s1']['entity_id'])
            best_threshold, best_f05 = sweep_threshold(
                val_pairs, val_probs, data['val_gt'], all_s1_ids=val_s1_ids
            )
            timing_dict['Stage 6: Threshold Tuning'] = time.time() - t0_stage6
        
        print("\n[cache] Saving trained model and threshold to cache...")
        joblib.dump({
            'matcher': matcher,
            'best_threshold': best_threshold,
            'best_f05': best_f05,
            'val_probs': val_probs,
            'val_s1_ids': val_s1_ids
        }, model_cache_path)

    with open(THRESHOLD_PATH, 'w', encoding='utf-8') as threshold_file:
        import json
        if isinstance(best_threshold, dict):
            json.dump(best_threshold, threshold_file)
        else:
            threshold_file.write(f"{best_threshold:.8f}\n")

    # ── 7. Diagnostics ────────────────────────────────────────────────────
    if len(val_pairs) > 0:
        print("\n" + "="*60)
        print("STAGE 7: Generating diagnostics")
        print("="*60)

        importance = matcher.feature_importance(FEATURE_NAMES)
        diagnostics = compute_diagnostics(
            pairs_df=val_pairs,
            probabilities=val_probs,
            labels=y_val,
            ground_truth=data['val_gt'],
            all_s1_ids=val_s1_ids,
            feature_names=FEATURE_NAMES,
            feature_importances=importance,
            threshold=best_threshold,
            blocking_recall=val_blocking_recall,
        )

    # ── 8. Free memory before test data ────────────────────────────────────
    print("\n[memory] Freeing training data from memory...")
    if 'train_pairs' in locals(): del train_pairs
    if 'val_pairs' in locals(): del val_pairs
    if 'X_train' in locals(): del X_train
    if 'X_val' in locals(): del X_val
    if 'data' in locals():
        for key in list(data.keys()):
            del data[key]
        del data
    import gc; gc.collect()

    if not skip_inference:
        run_test_inference(matcher, best_threshold)
        
    return {'val_f05': best_f05}

def run_test_inference(matcher=None, best_threshold=None):
    """Standalone module to run test inference without loading training memory."""
    import time, os, gc, joblib
    from config import PIPELINE_VERSION, SAMPLE_FRAC, OUTPUT_DIR
    from blocking.blocker import generate_candidates
    from features.similarity import extract_features
    from postprocessing.threshold import format_and_save_output
    # Local functions _run_validation and _preprocess_all are already in scope
    
    sample_key = f"{SAMPLE_FRAC:.1f}"
    
    # If resuming a crashed pipeline, we load the cached models
    if matcher is None or best_threshold is None:
        model_cache_path = os.path.join(CACHE_DIR, f'model_cache_{PIPELINE_VERSION}_{sample_key}.pkl')
        model_data = joblib.load(model_cache_path)
        matcher = model_data['matcher']
        best_threshold = model_data['best_threshold']
        
    timing_dict = {}

    print("\n" + "="*60)
    print("STAGE 8-12: Iterative Test Inference")
    print("="*60)
    
    t0_stage8 = time.time()
    test_data_path = os.path.join(CACHE_DIR, 'test_data.pkl')
    if os.path.exists(test_data_path):
        print("[cache] Loading preprocessed test data from cache...")
        test_data = joblib.load(test_data_path)
    else:
        test_data = load_test_data()
        test_data = _preprocess_all(test_data, ['test'])
        joblib.dump(test_data, test_data_path)
        
    tsv_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    # V4: Use stable tag instead of formatting dict as float (fixes TypeError)
    if isinstance(best_threshold, dict):
        threshold_tag = "v4_percountry"
    else:
        threshold_tag = f"{best_threshold:.3f}"
    progress_path = os.path.join(
        CACHE_DIR,
        f'test_progress_{PIPELINE_VERSION}_{sample_key}_threshold_{threshold_tag}.txt',
    )
    
    completed_countries = set()
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            completed_countries = set(f.read().splitlines())
            
    countries = sorted(test_data['test_s1']['country'].unique())
    total_matching_df_parts = []
    
    for country in countries:
        if country in completed_countries:
            print(f"\n[inference] Skipping {country} (Already completed)")
            continue
            
        print(f"\n[inference] --- Processing {country} ---")
        country_pairs = generate_candidates(
            test_data['test_s1'], test_data['test_s2'], test_data['test_s3'],
            target_country=country,
            cache_prefix=f'test_{PIPELINE_VERSION}'
        )
        
        probs_cache_path = os.path.join(CACHE_DIR, f'test_probs_{PIPELINE_VERSION}_{sample_key}_{threshold_tag}_{country}.pkl')
        if os.path.exists(probs_cache_path):
            print("\n[cache] Loading precomputed probabilities...")
            test_probs = joblib.load(probs_cache_path)
        else:
            print("\n[features] Extracting features ...")
            X_test = extract_features(
                country_pairs, test_data['test_s1'], test_data['test_s2'], test_data['test_s3']
            )
            test_probs = matcher.predict_proba(X_test)
            joblib.dump(test_probs, probs_cache_path)
            
            # Free memory early
            del X_test
            import gc
            gc.collect()
        
        # (Reranker bypassed - V4 Ensemble)
        # --------------------------
        
        test_s1_ids = test_data['test_s1'][test_data['test_s1']['country'] == country]['entity_id']
        append_mode = os.path.exists(tsv_path) and len(completed_countries) > 0
        
        preds_df = format_and_save_output(
            country_pairs, test_probs, best_threshold, test_s1_ids,
            append_mode=append_mode
        )
        
        completed_countries.add(country)
        with open(progress_path, 'a') as f:
            f.write(country + '\n')
            
        del country_pairs, test_probs, preds_df
        gc.collect()
    timing_dict['Stage 8-12: Test Inference'] = time.time() - t0_stage8

    print("\n" + "="*60)
    print("STAGE 10: Validating submission")
    print("="*60)
    
    # Save model
    matcher.save()
    
    _run_validation()

    print(f"\n{'='*60}")
    print("TIMING SUMMARY:")
    for stage_name, duration in timing_dict.items():
        print(f"  {stage_name:.<45} {duration/60:>6.1f} min")
    print(f"{'='*60}")
    print("PIPELINE COMPLETE")
    best_t_str = str({k: f"{v:.3f}" for k, v in best_threshold.items()}) if isinstance(best_threshold, dict) else f"{best_threshold:.3f}"
    print(f"  Val F0.5: {best_f05:.4f} @ threshold {best_t_str}")
    print(f"  Blocking recall (train): {train_blocking_recall:.4f}")
    print(f"  Blocking recall (val): {val_blocking_recall:.4f}")
    print(f"{'='*60}")



def run_cv():
    """5-fold stratified group CV on training data."""
    from sklearn.model_selection import StratifiedGroupKFold
    from evaluation.metrics import f05_macro as compute_f05

    print("\n" + "="*60)
    print("CROSS-VALIDATION MODE")
    print("="*60)

    # Load full training data (no val split needed for CV)
    data = load_train_data(sample_frac=SAMPLE_FRAC, val_frac=0.0)

    # For CV mode we need train_s1 to be all S1, etc.
    # But our load function always splits — let's load with val_frac=0
    # Actually, let's use full data by reloading with val_frac very small
    # and recombining. For now, let's just use the train split from normal loading.

    # Re-load with standard split
    data = load_train_data(sample_frac=SAMPLE_FRAC)

    # Combine train + val back for CV
    all_s1 = pd.concat([data['train_s1'], data['val_s1']], ignore_index=True)
    all_s2 = pd.concat([data['train_s2'], data['val_s2']], ignore_index=True)
    all_s3 = pd.concat([data['train_s3'], data['val_s3']], ignore_index=True)
    all_gt = {**data['train_gt'], **data['val_gt']}

    # Preprocess
    combined = {'all_s1': all_s1, 'all_s2': all_s2, 'all_s3': all_s3}
    for key in combined:
        print(f"\n[preprocess] Cleaning {key} ({len(combined[key]):,} records) ...")
        combined[key] = preprocess_dataframe(combined[key])
        is_s1 = key.endswith('s1')
        print(f"[preprocess] Transliterating {key} (is_s1={is_s1}) ...")
        combined[key] = apply_transliteration(combined[key], is_s1=is_s1)

    all_s1 = combined['all_s1']
    all_s2 = combined['all_s2']
    all_s3 = combined['all_s3']

    # Set up CV
    s1_ids = all_s1['entity_id'].values
    s1_countries = all_s1['country'].values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(all_s1, s1_countries, s1_ids)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*60}")

        fold_s1_train = all_s1.iloc[train_idx]
        fold_s1_val = all_s1.iloc[val_idx]
        fold_s1_train_ids = set(fold_s1_train['entity_id'])
        fold_s1_val_ids = set(fold_s1_val['entity_id'])

        # Split GT
        fold_train_gt = {k: v for k, v in all_gt.items() if k in fold_s1_train_ids}
        fold_val_gt = {k: v for k, v in all_gt.items() if k in fold_s1_val_ids}

        # Split S2/S3 by matched IDs
        train_rev = {}
        for s1_id, mids in fold_train_gt.items():
            for mid in mids:
                train_rev[mid] = s1_id
        val_rev = {}
        for s1_id, mids in fold_val_gt.items():
            for mid in mids:
                val_rev[mid] = s1_id

        train_matched = set(train_rev.keys())
        val_matched = set(val_rev.keys())

        fold_s2_train = all_s2[all_s2['entity_id'].isin(train_matched) |
                               ~all_s2['entity_id'].isin(train_matched | val_matched)]
        fold_s2_val = all_s2[all_s2['entity_id'].isin(val_matched)]
        fold_s3_train = all_s3[all_s3['entity_id'].isin(train_matched) |
                               ~all_s3['entity_id'].isin(train_matched | val_matched)]
        fold_s3_val = all_s3[all_s3['entity_id'].isin(val_matched)]

        # Block
        train_pairs = generate_candidates(fold_s1_train, fold_s2_train, fold_s3_train)
        val_pairs = generate_candidates(fold_s1_val, fold_s2_val, fold_s3_val)

        # Features
        X_train = extract_features(train_pairs, fold_s1_train, fold_s2_train, fold_s3_train)
        y_train = generate_labels(train_pairs, fold_train_gt)
        X_val = extract_features(val_pairs, fold_s1_val, fold_s2_val, fold_s3_val)
        y_val = generate_labels(val_pairs, fold_val_gt)

        # Train
        matcher = EntityMatcher()
        matcher.train(X_train, y_train, X_val, y_val, feature_names=FEATURE_NAMES)

        # Predict + threshold
        if len(val_pairs) == 0:
            print("\n[threshold] 100% Run Detected (val_frac=0.0): Skipping validation sweep.")
            print("[threshold] Loading pre-tuned 80/20 thresholds...")
            import json
            with open(THRESHOLD_PATH, "r") as f:
                try:
                    best_threshold = json.load(f)
                except:
                    best_threshold = float(f.read().strip())
            best_f05 = 0.0
            val_probs = np.array([])
            val_s1_ids = set()
            
            print("\n[cache] Saving trained model and threshold to cache...")
            joblib.dump({
                "matcher": matcher,
                "best_threshold": best_threshold,
                "best_f05": best_f05,
                "val_probs": val_probs,
                "val_s1_ids": val_s1_ids
            }, model_cache_path)
        else:
            val_probs = matcher.predict_proba(X_val)
        
        # (Reranker bypassed - V4)
        
        best_t, best_f05 = sweep_threshold(val_pairs, val_probs, fold_val_gt, fold_s1_val_ids)

        fold_scores.append(best_f05)
        best_t_str = str({k: f"{v:.3f}" for k, v in best_t.items()}) if isinstance(best_t, dict) else f"{best_t:.3f}"
        print(f"  Fold {fold_idx + 1} F0.5: {best_f05:.4f} @ threshold {best_t_str}")

        gc.collect()

    mean_f05 = np.mean(fold_scores)
    std_f05 = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"CV RESULTS: F0.5 = {mean_f05:.4f} +- {std_f05:.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"{'='*60}")

    return {'cv_mean': mean_f05, 'cv_std': std_f05, 'fold_scores': fold_scores}


def _run_validation():
    """Run the official validate_submission.py script."""
    matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    test_dir = str(TEST_DIR)

    cmd = [
        sys.executable, str(VALIDATE_SCRIPT),
        '--matching', matching_path,
        # '--candidate', candidate_path, # Skipped due to MemoryError on 250M+ candidates
        '--test-dir', test_dir,
    ]

    print(f"[validate] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("[validate] [FAIL] VALIDATION FAILED")
    else:
        print("[validate] [PASS] VALIDATION PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "cv"],
        default="train",
        help="Pipeline mode (default: train)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip test data inference (useful for fast CV iteration)"
    )
    parser.add_argument(
        "--loco-val",
        type=str,
        default=None,
        help="Country to use for LOCO (Leave-One-Country-Out) validation (e.g. 'India')"
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train(skip_inference=args.skip_inference, loco_country=args.loco_val)
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "cv":
        run_cv()


