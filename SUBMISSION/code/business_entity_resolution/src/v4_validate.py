"""
V4 Pure-Ensemble Validation — Reproduce the 0.8878 F0.5 from clean caches.

This script does NOT use model_cache_1.pkl or val_probs_reranked_1.pkl.
It loads the shared LightGBM + XGBoost models, generates fresh ensemble
probabilities, and sweeps per-country thresholds.

Usage:
    python src/v4_validate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import gc
from datetime import datetime, timezone, timedelta

import numpy as np
import joblib

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_FRAC, BLOCKING_TOP_K, TFIDF_NGRAM_RANGE
from postprocessing.threshold import sweep_threshold
from models.matcher import EntityMatcher

# ── Paths ────────────────────────────────────────────────────────────────────
SHARED_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'
)
V4_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache_v4'
)
os.makedirs(V4_CACHE, exist_ok=True)

SAMPLE_KEY = f"{SAMPLE_FRAC:g}"


def main():
    total_start = time.time()

    # ── 1. Load pure ensemble models (shared, version-agnostic) ──────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Loading pure ensemble models")
    print("=" * 60)

    lgb_path = os.path.join(SHARED_CACHE, 'lgb_model.pkl')
    xgb_path = os.path.join(SHARED_CACHE, 'xgb_model.pkl')

    if not os.path.exists(lgb_path) or not os.path.exists(xgb_path):
        print("[FATAL] LightGBM or XGBoost model not found in shared cache.")
        print(f"  Expected: {lgb_path}")
        print(f"  Expected: {xgb_path}")
        sys.exit(1)

    matcher = EntityMatcher()
    matcher.lgb_model = joblib.load(lgb_path)
    matcher.xgb_model = joblib.load(xgb_path)
    print(f"[v4] Loaded LightGBM from {lgb_path}")
    print(f"[v4] Loaded XGBoost from {xgb_path}")

    # Verify: NO reference to model_cache_1.pkl or val_probs_reranked
    v3_cache = os.path.join(SHARED_CACHE, f'model_cache_{SAMPLE_KEY}.pkl')
    v3_reranked = os.path.join(SHARED_CACHE, f'val_probs_reranked_{SAMPLE_KEY}.pkl')
    print(f"\n[v4] V3 contamination check:")
    print(f"  model_cache_1.pkl exists: {os.path.exists(v3_cache)} (NOT loaded)")
    print(f"  val_probs_reranked_1.pkl exists: {os.path.exists(v3_reranked)} (NOT loaded)")

    # ── 2. Load validation features (mmap, zero RAM) ─────────────────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Loading validation features (mmap)")
    print("=" * 60)

    val_feat_path = os.path.join(SHARED_CACHE, f'val_feat_{SAMPLE_KEY}.pkl')
    if not os.path.exists(val_feat_path):
        print(f"[FATAL] val_feat_{SAMPLE_KEY}.pkl not found.")
        sys.exit(1)

    t0 = time.time()
    X_val, y_val = joblib.load(val_feat_path, mmap_mode='r')
    print(f"[v4] Loaded val features: X_val={X_val.shape}, y_val={y_val.shape}  ({time.time()-t0:.1f}s)")

    # ── 3. Load validation pairs and blocking recall ─────────────────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Loading validation pairs")
    print("=" * 60)

    val_pairs_path = os.path.join(SHARED_CACHE, f'val_pairs_{SAMPLE_KEY}.pkl')
    if not os.path.exists(val_pairs_path):
        print(f"[FATAL] val_pairs_{SAMPLE_KEY}.pkl not found.")
        sys.exit(1)

    t0 = time.time()
    val_pairs, val_blocking_recall = joblib.load(val_pairs_path)
    print(f"[v4] Loaded val pairs: {len(val_pairs):,} rows  ({time.time()-t0:.1f}s)")
    print(f"[v4] Val blocking recall: {val_blocking_recall:.6f}")
    print(f"[v4] Val pairs columns: {list(val_pairs.columns)}")
    if 'country' in val_pairs.columns:
        countries = val_pairs['country'].unique()
        for c in sorted(countries):
            print(f"  {c}: {(val_pairs['country'] == c).sum():,} pairs")

    # ── 4. Load ground truth ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Loading ground truth")
    print("=" * 60)

    train_data_path = os.path.join(SHARED_CACHE, f'train_data_{SAMPLE_KEY}.pkl')
    if not os.path.exists(train_data_path):
        print(f"[FATAL] train_data_{SAMPLE_KEY}.pkl not found.")
        sys.exit(1)

    t0 = time.time()
    data = joblib.load(train_data_path)
    val_gt = data['val_gt']
    val_s1_ids = set(data['val_s1']['entity_id'])
    val_countries_s1 = data['val_s1'][['entity_id', 'country']].copy()
    del data
    gc.collect()
    print(f"[v4] Ground truth: {len(val_gt)} S1 entities with matches  ({time.time()-t0:.1f}s)")
    print(f"[v4] Total val S1 entities: {len(val_s1_ids):,}")

    # ── 5. Generate fresh pure-ensemble probabilities ────────────────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Computing fresh ensemble probabilities")
    print("=" * 60)
    print("[v4] Method: (LightGBM_prob + XGBoost_prob) / 2.0")
    print("[v4] NO reranker. NO penalty. Pure ensemble only.")

    t0 = time.time()
    val_probs = matcher.predict_proba(X_val)
    prob_time = time.time() - t0
    print(f"[v4] Probabilities computed: {len(val_probs):,} values  ({prob_time:.1f}s)")
    print(f"  min={val_probs.min():.8f}  max={val_probs.max():.8f}  mean={val_probs.mean():.8f}")

    # Free features
    del X_val, y_val
    gc.collect()

    # ── 6. Sweep per-country thresholds ──────────────────────────────────
    print("\n" + "=" * 60)
    print("V4 VALIDATION: Threshold sweep (per-country)")
    print("=" * 60)

    t0 = time.time()
    best_threshold, best_f05 = sweep_threshold(
        val_pairs, val_probs, val_gt, all_s1_ids=val_s1_ids
    )
    sweep_time = time.time() - t0
    print(f"\n[v4] Threshold sweep took {sweep_time:.1f}s")

    # ── 7. Report results ────────────────────────────────────────────────
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("V4 VALIDATION RESULTS")
    print("=" * 60)
    print(f"  F0.5 (macro):          {best_f05:.6f}")
    if isinstance(best_threshold, dict):
        for country, t in sorted(best_threshold.items()):
            print(f"  Threshold ({country}):      {t:.4f}")
    else:
        print(f"  Threshold (global):    {best_threshold:.4f}")
    print(f"  Blocking recall (val): {val_blocking_recall:.6f}")
    print(f"  Val pairs:             {len(val_pairs):,}")
    print(f"  Val S1 entities:       {len(val_s1_ids):,}")
    print(f"  Ensemble method:       (LGB + XGB) / 2")
    print(f"  Reranker:              NONE")
    print(f"  Total time:            {total_time:.1f}s")
    print("=" * 60)

    # ── 8. Save V4 cache and manifest ────────────────────────────────────
    print("\n[v4] Saving V4 model cache...")
    model_cache_v4_path = os.path.join(V4_CACHE, 'model_cache_v4.pkl')
    joblib.dump({
        'matcher': matcher,
        'best_threshold': best_threshold,
        'best_f05': best_f05,
        'val_probs': val_probs,
        'val_s1_ids': val_s1_ids,
        'val_blocking_recall': val_blocking_recall,
    }, model_cache_v4_path)
    print(f"  -> {model_cache_v4_path}")

    # Write manifest
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')

    manifest = {
        'experiment_id': 'v4_pure_ensemble',
        'date': now_ist,
        'val_f05': float(best_f05),
        'thresholds': {k: float(v) for k, v in best_threshold.items()} if isinstance(best_threshold, dict) else float(best_threshold),
        'val_blocking_recall': float(val_blocking_recall),
        'val_pairs_count': len(val_pairs),
        'val_s1_count': len(val_s1_ids),
        'models_used': {
            'lgb': os.path.basename(lgb_path),
            'xgb': os.path.basename(xgb_path),
        },
        'ensemble_method': 'average',
        'reranker': 'none',
        'v3_artifacts_used': False,
        'config': {
            'SAMPLE_FRAC': SAMPLE_FRAC,
            'BLOCKING_TOP_K': BLOCKING_TOP_K,
            'TFIDF_NGRAM_RANGE': list(TFIDF_NGRAM_RANGE),
        },
        'prob_stats': {
            'min': float(val_probs.min()),
            'max': float(val_probs.max()),
            'mean': float(val_probs.mean()),
        },
        'timing': {
            'prob_generation_s': round(prob_time, 1),
            'threshold_sweep_s': round(sweep_time, 1),
            'total_s': round(total_time, 1),
        },
    }

    manifest_path = os.path.join(V4_CACHE, 'v4_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[v4] Manifest written to {manifest_path}")

    print(f"\n[v4] V4 validation complete. F0.5 = {best_f05:.6f}")
    return best_f05, best_threshold


if __name__ == '__main__':
    main()
