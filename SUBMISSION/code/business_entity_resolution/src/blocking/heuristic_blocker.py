"""
Heuristic Blockers - Fast CPU streams to recover Semantic misses.
Provides:
1. Targeted TF-IDF for low-confidence 'orphan' records.
2. Rare-Token Hash blocking for instant exact matching.
"""

import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def _generate_tfidf_candidates(s1_df: pd.DataFrame, s23_df: pd.DataFrame, top_k: int = 5, ngram_range: tuple = (3,3)) -> pd.DataFrame:
    """Targeted chunked TF-IDF designed exclusively for a small subset of S1 records."""
    if len(s1_df) == 0 or len(s23_df) == 0:
        return pd.DataFrame()
        
    print(f"    [TF-IDF] Fitting Vectorizer on {len(s23_df)} S2/S3 records...")
    t0 = time.time()
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, max_features=100000, min_df=2)
    s23_matrix = vec.fit_transform(s23_df['name_address'].fillna(''))
    
    print(f"    [TF-IDF] Transforming {len(s1_df)} Orphan S1 records...")
    s1_matrix = vec.transform(s1_df['name_address'].fillna(''))
    print(f"    [TF-IDF] Vectorization took {time.time()-t0:.1f}s")
    
    print(f"    [TF-IDF] Executing sparse dot product (chunked)...")
    t1 = time.time()
    
    chunk_size = 100  # Strict limit to prevent MemoryError on dense conversions
    s1_ids = s1_df['entity_id'].values
    s23_ids = s23_df['entity_id'].values
    s23_sources = s23_df['source'].values if 'source' in s23_df.columns else np.array(['S2' if str(x).startswith('S2') else 'S3' for x in s23_ids])
    country = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
    
    all_s1_idx = []
    all_s23_idx = []
    all_scores = []
    
    for i in range(0, s1_matrix.shape[0], chunk_size):
        chunk = s1_matrix[i:i+chunk_size]
        sim_scores = chunk.dot(s23_matrix.T).toarray()
        
        for row_idx in range(sim_scores.shape[0]):
            row_scores = sim_scores[row_idx]
            k = min(top_k, len(row_scores))
            if k == 0: continue
            
            top_k_idx = np.argpartition(row_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(row_scores[top_k_idx])][::-1]
            
            # Prune very weak matches
            valid = top_k_idx[row_scores[top_k_idx] > 0.10]
            if len(valid) == 0: continue
            
            abs_s1_idx = i + row_idx
            all_s1_idx.extend([abs_s1_idx] * len(valid))
            all_s23_idx.extend(valid)
            all_scores.extend(row_scores[valid])
            
    print(f"    [TF-IDF] Dot product & Top-K took {time.time()-t1:.1f}s")
    
    return pd.DataFrame({
        's1_id': s1_ids[all_s1_idx],
        's2s3_id': s23_ids[all_s23_idx],
        'source': s23_sources[all_s23_idx],
        'country': country,
        'semantic_score': all_scores
    })


def _generate_rare_token_candidates(s1_df: pd.DataFrame, s23_df: pd.DataFrame, max_freq: int = 15) -> pd.DataFrame:
    """O(N) Hash Blocking on rare tokens to find exact subsets (phone numbers, weird acronyms)."""
    t0 = time.time()
    if len(s1_df) == 0 or len(s23_df) == 0:
        return pd.DataFrame()
        
    def get_tokens(df, id_col):
        texts = df['name_address'].fillna('').str.lower()
        df_tok = pd.DataFrame({id_col: df['entity_id'], 'text': texts})
        df_tok['token'] = df_tok['text'].apply(lambda x: re.findall(r'\b[a-z0-9]{3,}\b', x))
        return df_tok.explode('token')[[id_col, 'token']]
        
    s1_tok = get_tokens(s1_df, 's1_id').dropna()
    s23_tok = get_tokens(s23_df, 's2s3_id').dropna()
    
    # Cap frequency to prevent memory explosion
    token_counts = s1_tok['token'].value_counts()
    rare_tokens = set(token_counts[token_counts <= max_freq].index)
    
    s1_rare = s1_tok[s1_tok['token'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['token'].isin(rare_tokens)]
    
    merged = pd.merge(s1_rare, s23_rare, on='token', how='inner')
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id']).copy()
    
    if len(merged) > 0:
        if 'source' in s23_df.columns:
            source_map = s23_df.set_index('entity_id')['source'].to_dict()
            merged['source'] = merged['s2s3_id'].map(source_map)
        else:
            merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
            
        merged['country'] = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
        merged['semantic_score'] = 0.60  # Base confidence so it survives downstream pruning
    
    print(f"    [Rare-Token] Found {len(merged):,} candidates in {time.time()-t0:.1f}s")
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]
