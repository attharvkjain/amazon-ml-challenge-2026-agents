import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_FRAC
from blocking.blocker import compute_blocking_recall
from blocking.heuristic_blocker import _generate_tfidf_candidates, _generate_rare_token_candidates, _generate_phonetic_candidates, _generate_bigram_candidates

SHARED_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"

def main():
    print("=" * 60)
    print("V6.1 TEST: ORPHAN-TARGETED BI-GRAM & PHONETIC BLOCKING")
    print("=" * 60)
    
    print("[1] Loading existing Semantic candidates and Ground Truth...")
    train_data_path = os.path.join(SHARED_CACHE, f'train_data_{SAMPLE_KEY}.pkl')
    data = joblib.load(train_data_path)
    s1 = data['val_s1']
    s2 = data['val_s2']
    s3 = data['val_s3']
    s23 = pd.concat([s2, s3])
    s23['source'] = s23['entity_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
    val_gt = data['val_gt']
    
    val_pairs_path = os.path.join(SHARED_CACHE, f'val_pairs_{SAMPLE_KEY}.pkl')
    semantic_cands = joblib.load(val_pairs_path)[0]
    
    base_recall = compute_blocking_recall(semantic_cands, val_gt)
    
    print("\n[2] Identifying Orphans...")
    # S1 Orphans
    s1_max = semantic_cands.groupby('s1_id')['semantic_score'].max()
    orphan_s1_ids = set(s1_max[s1_max < 0.75].index).union(set(s1['entity_id']) - set(semantic_cands['s1_id']))
    
    # S23 Orphans
    s23_max = semantic_cands.groupby('s2s3_id')['semantic_score'].max()
    orphan_s23_ids = set(s23_max[s23_max < 0.85].index).union(set(s23['entity_id']) - set(semantic_cands['s2s3_id']))
    
    print(f"    S1 Orphans: {len(orphan_s1_ids):,}")
    print(f"    S23 Orphans: {len(orphan_s23_ids):,}")
    
    new_cands_list = []
    countries = s1['country'].unique()
    for c in countries:
        c_s1 = s1[s1['country'] == c]
        c_s23 = s23[s23['country'] == c]
        c_s1_orphans = c_s1[c_s1['entity_id'].isin(orphan_s1_ids)]
        c_s23_orphans = c_s23[c_s23['entity_id'].isin(orphan_s23_ids)]
        
        if len(c_s1_orphans) > 0:
            new_cands_list.append(_generate_tfidf_candidates(c_s1_orphans, c_s23, top_k=5))
            
        new_cands_list.append(_generate_rare_token_candidates(c_s1, c_s23, max_freq=15))
        
        if len(c_s23_orphans) > 0:
            new_cands_list.append(_generate_bigram_candidates(c_s1, c_s23_orphans, max_freq=100))
            new_cands_list.append(_generate_phonetic_candidates(c_s1, c_s23_orphans, max_freq=50))
            
    print("\n[4] Unioning and computing final recall...")
    combined = pd.concat([semantic_cands] + new_cands_list).drop_duplicates(subset=['s1_id', 's2s3_id'])
    
    print(f"    Final Candidate Volume: {len(combined):,} (Added {len(combined)-len(semantic_cands):,})")
    final_recall = compute_blocking_recall(combined, val_gt)
    
    print("=" * 60)
    print(f"RESULT: Recall improved from {base_recall*100:.2f}% to {final_recall*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
