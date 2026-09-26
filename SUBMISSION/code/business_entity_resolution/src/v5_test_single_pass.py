"""
V5 Phase B Diagnostic: Single-Pass Conditional Retrieval Test

Loads the existing semantic candidates, isolates orphans using the 
cosine similarity heuristic (< 0.75), runs targeted TF-IDF, runs O(N) hash 
blocking, unions the results, and calculates the massive recall gain.
"""

import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_FRAC
from blocking.blocker import compute_blocking_recall

SHARED_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"

def _generate_tfidf_candidates(s1_df, s23_df, top_k=5, ngram_range=(3,3)):
    """Targeted chunked TF-IDF for orphans only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print(f"    [TF-IDF] Fitting Vectorizer on {len(s23_df)} S2/S3 records...")
    t0 = time.time()
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, max_features=100000, min_df=2)
    s23_matrix = vec.fit_transform(s23_df['name_address'].fillna(''))
    
    print(f"    [TF-IDF] Transforming {len(s1_df)} Orphan S1 records...")
    s1_matrix = vec.transform(s1_df['name_address'].fillna(''))
    print(f"    [TF-IDF] Vectorization took {time.time()-t0:.1f}s")
    
    print(f"    [TF-IDF] Executing sparse dot product chunked...")
    t1 = time.time()
    
    chunk_size = 100
    s1_ids = s1_df['entity_id'].values
    s23_ids = s23_df['entity_id'].values
    s23_sources = s23_df['source'].values
    country = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
    
    all_s1_idx = []
    all_s23_idx = []
    all_scores = []
    
    for i in range(0, s1_matrix.shape[0], chunk_size):
        chunk = s1_matrix[i:i+chunk_size]
        sim_scores = chunk.dot(s23_matrix.T).toarray()
        
        # Get top K for each query in chunk
        for row_idx in range(sim_scores.shape[0]):
            row_scores = sim_scores[row_idx]
            k = min(top_k, len(row_scores))
            # argpartition is faster than argsort for top-k
            top_k_idx = np.argpartition(row_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(row_scores[top_k_idx])][::-1]
            
            # Prune absolute garbage matches
            valid = top_k_idx[row_scores[top_k_idx] > 0.10]
            
            abs_s1_idx = i + row_idx
            all_s1_idx.extend([abs_s1_idx] * len(valid))
            all_s23_idx.extend(valid)
            all_scores.extend(row_scores[valid])
            
    print(f"    [TF-IDF] Dot product & Top-K took {time.time()-t1:.1f}s")
    
    df = pd.DataFrame({
        's1_id': s1_ids[all_s1_idx],
        's2s3_id': s23_ids[all_s23_idx],
        'source': s23_sources[all_s23_idx],
        'country': country,
        'semantic_score': all_scores  # Reusing column name for pipeline compatibility
    })
    return df


def _generate_rare_token_candidates(s1_df, s23_df, max_freq=15):
    """O(N) Hash Blocking on rare tokens."""
    t0 = time.time()
    
    def get_tokens(df, id_col):
        # Extract alphanumeric words >= 3 chars
        texts = df['name_address'].fillna('').str.lower()
        # Find all matches and explode
        df_tok = pd.DataFrame({id_col: df['entity_id'], 'text': texts})
        df_tok['token'] = df_tok['text'].apply(lambda x: re.findall(r'\b[a-z0-9]{3,}\b', x))
        return df_tok.explode('token')[[id_col, 'token']]
        
    s1_tok = get_tokens(s1_df, 's1_id').dropna()
    s23_tok = get_tokens(s23_df, 's2s3_id').dropna()
    
    # Cap frequency using S1 tokens
    token_counts = s1_tok['token'].value_counts()
    rare_tokens = token_counts[token_counts <= max_freq].index
    
    s1_rare = s1_tok[s1_tok['token'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['token'].isin(rare_tokens)]
    
    # Fast Hash Join
    merged = pd.merge(s1_rare, s23_rare, on='token', how='inner')
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id'])
    
    merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if x.startswith('S2') else 'S3')
    merged['country'] = s1_df['country'].iloc[0]
    merged['semantic_score'] = 0.60 # Dummy score to pass downstream filters
    
    print(f"    [Rare-Token] Found {len(merged):,} candidates in {time.time()-t0:.1f}s")
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]


def main():
    print("=" * 60)
    print("V5 TEST: SINGLE-PASS CONDITIONAL BLOCKING")
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
    
    print(f"    Base Semantic Candidates: {len(semantic_cands):,}")
    base_recall = compute_blocking_recall(semantic_cands, val_gt)
    print(f"    Base Recall: {base_recall*100:.2f}%")
    
    # ── HEURISTIC ──
    # If the max semantic_score for an S1 is < 0.75, it's an orphan.
    # Note: 'semantic_score' in val_pairs_1.pkl is exactly the cosine similarity.
    print("\n[2] Applying Heuristic (Orphan Detection)...")
    max_scores = semantic_cands.groupby('s1_id')['semantic_score'].max()
    orphan_ids = set(max_scores[max_scores < 0.75].index)
    
    # Add totally missed S1 entities
    missed_ids = set(s1['entity_id']) - set(semantic_cands['s1_id'])
    all_orphan_ids = orphan_ids.union(missed_ids)
    
    print(f"    Orphans identified: {len(all_orphan_ids):,} / {len(s1):,} S1 entities ({(len(all_orphan_ids)/len(s1))*100:.1f}%)")
    
    new_cands_list = []
    
    print("\n[3] Running Targeted Streams (Per Country)...")
    countries = s1['country'].unique()
    for c in countries:
        print(f"\n  -- Country: {c} --")
        c_s1 = s1[s1['country'] == c].copy()
        c_s23 = s23[s23['country'] == c].copy()
        c_orphans = c_s1[c_s1['entity_id'].isin(all_orphan_ids)].copy()
        
        if len(c_orphans) > 0:
            tfidf_df = _generate_tfidf_candidates(c_orphans, c_s23, top_k=5)
            new_cands_list.append(tfidf_df)
            
        rare_df = _generate_rare_token_candidates(c_s1, c_s23, max_freq=15)
        new_cands_list.append(rare_df)
        
    print("\n[4] Unioning and computing final recall...")
    all_new_cands = pd.concat(new_cands_list)
    combined_cands = pd.concat([semantic_cands, all_new_cands])
    combined_cands = combined_cands.drop_duplicates(subset=['s1_id', 's2s3_id'])
    
    print(f"    Final Candidate Volume: {len(combined_cands):,} (Added {len(combined_cands)-len(semantic_cands):,})")
    
    final_recall = compute_blocking_recall(combined_cands, val_gt)
    print("=" * 60)
    print(f"RESULT: Recall improved from {base_recall*100:.2f}% to {final_recall*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
