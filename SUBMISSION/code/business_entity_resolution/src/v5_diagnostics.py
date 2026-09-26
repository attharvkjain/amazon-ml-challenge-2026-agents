"""
V5 Phase A Diagnostics - Categorize Validation False Negatives.

Analyzes the exact cause of every false negative using V4 pure-ensemble
predictions and the original validation ground truth.

Categories:
1. Not Retrieved (Blocker missed it)
2. Threshold Rejection (Model scored below country threshold)
3. 1:1 Conflict (Another S1 stole the S2/S3 record)

Outputs a summary to console and a detailed CSV for manual inspection.
"""
from __future__ import annotations

import os
import sys
import gc
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_FRAC
from postprocessing.threshold import _apply_threshold_and_constraint

# ── Paths ────────────────────────────────────────────────────────────────────
SHARED_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
V4_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache_v4')
V5_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache_v5')
os.makedirs(V5_CACHE, exist_ok=True)

SAMPLE_KEY = f"{SAMPLE_FRAC:g}"

def main():
    print("\n" + "=" * 60)
    print("V5 DIAGNOSTICS: FALSE NEGATIVE CATEGORIZATION")
    print("=" * 60)

    # 1. Load Caches
    print("[1/5] Loading V4 predictions and shared caches...")
    v4_cache_path = os.path.join(V4_CACHE, 'model_cache_v4.pkl')
    val_pairs_path = os.path.join(SHARED_CACHE, f'val_pairs_{SAMPLE_KEY}.pkl')
    train_data_path = os.path.join(SHARED_CACHE, f'train_data_{SAMPLE_KEY}.pkl')

    v4_cache = joblib.load(v4_cache_path)
    val_probs = v4_cache['val_probs']
    best_threshold = v4_cache['best_threshold']

    val_pairs = joblib.load(val_pairs_path)[0]
    val_pairs['prob'] = val_probs

    data = joblib.load(train_data_path)
    val_gt = data['val_gt']
    
    # 2. Build GT DataFrame
    print("[2/5] Constructing Ground Truth matrix...")
    gt_records = []
    for s1_id, matches in val_gt.items():
        for match_id in matches:
            gt_records.append({'s1_id': s1_id, 's2s3_id': match_id})
    gt_df = pd.DataFrame(gt_records)
    print(f"  -> Total ground truth pairs: {len(gt_df):,}")

    # 3. Simulate Predictions
    print("[3/5] Simulating threshold and 1:1 constraints...")
    # This function returns a series of matched S2/S3 IDs for each S1
    preds_s = _apply_threshold_and_constraint(val_pairs, val_probs, best_threshold)
    
    pred_records = []
    for s1_id, matches in preds_s.items():
        for match_id in matches:
            pred_records.append({'s1_id': s1_id, 's2s3_id': match_id})
    pred_df = pd.DataFrame(pred_records)
    if len(pred_df) > 0:
        pred_df['is_tp'] = True
    else:
        pred_df = pd.DataFrame(columns=['s1_id', 's2s3_id', 'is_tp'])

    # 4. Merge and Categorize
    print("[4/5] Categorizing False Negatives...")
    
    # Merge GT with candidates to see what was retrieved
    analysis_df = gt_df.merge(
        val_pairs[['s1_id', 's2s3_id', 'prob', 'country']], 
        on=['s1_id', 's2s3_id'], 
        how='left'
    )
    
    # Merge with final predictions to identify True Positives
    analysis_df = analysis_df.merge(
        pred_df,
        on=['s1_id', 's2s3_id'],
        how='left'
    )
    analysis_df['is_tp'] = analysis_df['is_tp'].fillna(False)
    
    # Map country thresholds
    def get_thresh(c):
        if pd.isna(c) or not isinstance(best_threshold, dict):
            return best_threshold if not isinstance(best_threshold, dict) else np.mean(list(best_threshold.values()))
        return best_threshold.get(c, np.mean(list(best_threshold.values())))
        
    analysis_df['threshold'] = analysis_df['country'].apply(get_thresh)

    # Categorization Logic
    def categorize(row):
        if row['is_tp']:
            return 'True Positive'
        if pd.isna(row['prob']):
            return '1. Not Retrieved'
        if row['prob'] < row['threshold']:
            return '2. Threshold Rejection'
        return '3. 1:1 Conflict'

    analysis_df['Category'] = analysis_df.apply(categorize, axis=1)

    # 5. Extract Text for CSV Export
    print("[5/5] Extracting text for FN review export...")
    fn_df = analysis_df[analysis_df['Category'] != 'True Positive'].copy()
    
    # Get texts
    s1_text_map = data['val_s1'].set_index('entity_id')['name_address'].to_dict()
    s2_text_map = data['val_s2'].set_index('entity_id')['name_address'].to_dict()
    s3_text_map = data['val_s3'].set_index('entity_id')['name_address'].to_dict()
    
    def get_s2s3_text(eid):
        if eid in s2_text_map: return s2_text_map[eid]
        if eid in s3_text_map: return s3_text_map[eid]
        return ""

    fn_df['s1_text'] = fn_df['s1_id'].map(lambda x: s1_text_map.get(x, ""))
    fn_df['s2s3_text'] = fn_df['s2s3_id'].map(get_s2s3_text)
    
    # Fallback for country if it was not retrieved (NaN)
    s1_country_map = data['val_s1'].set_index('entity_id')['country'].to_dict()
    fn_df['country'] = fn_df['country'].fillna(fn_df['s1_id'].map(s1_country_map))

    # Save to CSV
    csv_path = os.path.join(V5_CACHE, 'fn_analysis.csv')
    fn_df.to_csv(csv_path, index=False)

    # ── Summary Report ───────────────────────────────────────────────
    total_gt = len(gt_df)
    total_fn = len(fn_df)
    
    cat_counts = fn_df['Category'].value_counts()
    not_retrieved = cat_counts.get('1. Not Retrieved', 0)
    thresh_rej = cat_counts.get('2. Threshold Rejection', 0)
    conflict_1v1 = cat_counts.get('3. 1:1 Conflict', 0)

    print("\n" + "=" * 60)
    print("V5 FALSE NEGATIVE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Total Ground Truth Pairs: {total_gt:,}")
    print(f"Total False Negatives:    {total_fn:,} ({(total_fn/total_gt)*100:.1f}%)")
    print("-" * 60)
    print("Breakdown of Failures:")
    print(f"  1. Not Retrieved (Blocker Miss): {not_retrieved:>8,} ({not_retrieved/total_fn*100:>5.1f}%)")
    print(f"  2. Threshold Rejection:          {thresh_rej:>8,} ({thresh_rej/total_fn*100:>5.1f}%)")
    print(f"  3. 1:1 Conflict (Stolen):        {conflict_1v1:>8,} ({conflict_1v1/total_fn*100:>5.1f}%)")
    print("-" * 60)
    
    print("By Country:")
    for c in fn_df['country'].unique():
        c_df = fn_df[fn_df['country'] == c]
        c_counts = c_df['Category'].value_counts()
        print(f"  {c}:")
        print(f"    Not Retrieved:        {c_counts.get('1. Not Retrieved', 0):,}")
        print(f"    Threshold Rejection:  {c_counts.get('2. Threshold Rejection', 0):,}")
        print(f"    1:1 Conflict:         {c_counts.get('3. 1:1 Conflict', 0):,}")
        
    print("=" * 60)
    print(f"Saved detailed CSV for manual review: {csv_path}")


if __name__ == '__main__':
    main()
