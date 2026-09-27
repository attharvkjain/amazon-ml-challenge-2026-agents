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
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer

def _get_budgeted_tokens(s1_tok, s23_tok, tok_col, budget):
    """
    Absolutely strict memory-bounded candidate generator.
    Prioritizes tokens that generate the FEWEST pairs, until the strict budget is reached.
    This entirely eliminates Cartesian combinatorial explosions regardless of dataset scale.
    """
    s1_counts = s1_tok[tok_col].value_counts()
    s23_counts = s23_tok[tok_col].value_counts()
    common = list(set(s1_counts.index).intersection(set(s23_counts.index)))
    if not common:
        return set()
        
    pairs = np.array([s1_counts[t] * s23_counts[t] for t in common])
    sorted_idx = np.argsort(pairs)
    cum_pairs = np.cumsum(pairs[sorted_idx])
    
    cutoff_idx = np.searchsorted(cum_pairs, budget, side='right')
    return {common[i] for i in sorted_idx[:cutoff_idx]}

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
    
    chunk_size = 10  # Strict limit to prevent MemoryError on dense conversions (10 * 4.7M = 370MB)
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
    
    budget = len(s1_df) * 5
    rare_tokens = _get_budgeted_tokens(s1_tok, s23_tok, 'token', budget)
    
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

def _generate_phonetic_candidates(s1_df: pd.DataFrame, s23_df: pd.DataFrame, max_freq: int = 50) -> pd.DataFrame:
    """O(N) Hash Blocking on phonetic tokens to recover transliteration errors and heavy misspellings."""
    t0 = time.time()
    if len(s1_df) == 0 or len(s23_df) == 0:
        return pd.DataFrame()
        
    def get_phones(df, id_col):
        texts = df['name_address'].fillna('').str.lower()
        df_tok = pd.DataFrame({id_col: df['entity_id'], 'text': texts})
        def extract(text):
            words = re.findall(r'\b[a-z]{4,}\b', text)
            phones = set()
            for w in words:
                p = jellyfish.metaphone(w)
                if len(p) >= 3: phones.add(p)
            return list(phones)
        df_tok['phone'] = df_tok['text'].apply(extract)
        return df_tok.explode('phone')[[id_col, 'phone']]
        
    s1_tok = get_phones(s1_df, 's1_id').dropna()
    s23_tok = get_phones(s23_df, 's2s3_id').dropna()
    
    budget = len(s1_df) * 10
    rare_tokens = _get_budgeted_tokens(s1_tok, s23_tok, 'phone', budget)
    
    s1_rare = s1_tok[s1_tok['phone'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['phone'].isin(rare_tokens)]
    
    merged = pd.merge(s1_rare, s23_rare, on='phone', how='inner')
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id']).copy()
    
    if len(merged) > 0:
        if 'source' in s23_df.columns:
            source_map = s23_df.set_index('entity_id')['source'].to_dict()
            merged['source'] = merged['s2s3_id'].map(source_map)
        else:
            merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
            
        merged['country'] = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
        merged['semantic_score'] = 0.60  # Base confidence
    
    print(f"    [Phonetic] Found {len(merged):,} candidates in {time.time()-t0:.1f}s")
    if len(merged) == 0: return pd.DataFrame()
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]

def _generate_bigram_candidates(s1_df: pd.DataFrame, s23_df: pd.DataFrame, max_freq: int = 100) -> pd.DataFrame:
    """O(N) Hash Blocking on consecutive word bi-grams. Excellent for catching aliases that share specific names."""
    t0 = time.time()
    if len(s1_df) == 0 or len(s23_df) == 0:
        return pd.DataFrame()
        
    def get_bigrams(df, id_col):
        texts = df['name_address'].fillna('').str.lower()
        df_tok = pd.DataFrame({id_col: df['entity_id'], 'text': texts})
        def extract(text):
            words = re.findall(r'\b[a-z0-9]+\b', text)
            if len(words) < 2: return []
            return [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
        df_tok['token'] = df_tok['text'].apply(extract)
        return df_tok.explode('token')[[id_col, 'token']]
        
    s1_tok = get_bigrams(s1_df, 's1_id').dropna()
    s23_tok = get_bigrams(s23_df, 's2s3_id').dropna()
    
    budget = len(s1_df) * 20
    rare_tokens = _get_budgeted_tokens(s1_tok, s23_tok, 'token', budget)
    
    s1_rare = s1_tok[s1_tok['token'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['token'].isin(rare_tokens)]
    
    merged = pd.merge(s1_rare, s23_rare, on='token', how='inner')
    
    # Critical Memory Optimization: Drop the massive string column BEFORE duplicate dropping
    merged = merged[['s1_id', 's2s3_id']]
    merged = merged.drop_duplicates()
    
    if len(merged) > 0:
        if 'source' in s23_df.columns:
            source_map = s23_df.set_index('entity_id')['source'].to_dict()
            merged['source'] = merged['s2s3_id'].map(source_map)
        else:
            merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
            
        merged['country'] = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
        merged['semantic_score'] = 0.60  # Base confidence
    
    print(f"    [Bi-gram] Found {len(merged):,} candidates in {time.time()-t0:.1f}s")
    if len(merged) == 0: return pd.DataFrame()
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]

def _generate_numeric_candidates(s1_df: pd.DataFrame, s23_df: pd.DataFrame, max_freq: int = 500) -> pd.DataFrame:
    """
    Highly targeted hash blocker for heavily transliterated S2/S3 text.
    Extracts purely alphanumeric tokens that contain digits (e.g. 501, 101, 4a, 704).
    Drops leading zeros to match 00501 to 501.
    Hashes these tokens across S1 and S23.
    """
    t0 = time.time()
    
    def extract_numeric_tokens(df):
        texts = df['name_address'].fillna('').str.lower()
        df_tok = pd.DataFrame({'id': df['entity_id'], 'text': texts})
        def extract(text):
            words = text.split()
            nums = [t.lstrip('0') for t in words if any(c.isdigit() for c in t)]
            return list(set(t for t in nums if t))
        df_tok['token'] = df_tok['text'].apply(extract)
        return df_tok.explode('token')[['id', 'token']].dropna()
        
    s1_tokens = extract_numeric_tokens(s1_df)
    s23_tokens = extract_numeric_tokens(s23_df)
    
    if len(s1_tokens) == 0 or len(s23_tokens) == 0:
        return pd.DataFrame()
        
    budget = len(s1_df) * 10
    rare_tokens = _get_budgeted_tokens(s1_tokens, s23_tokens, 'token', budget)
    
    s1_rare = s1_tokens[s1_tokens['token'].isin(rare_tokens)]
    s23_rare = s23_tokens[s23_tokens['token'].isin(rare_tokens)]
    
    if len(s1_rare) == 0 or len(s23_rare) == 0:
        return pd.DataFrame()
        
    merged = pd.merge(s1_rare, s23_rare, on='token', how='inner')
    merged.rename(columns={'id_x': 's1_id', 'id_y': 's2s3_id'}, inplace=True)
    
    merged = merged[['s1_id', 's2s3_id']].drop_duplicates()
    
    if len(merged) > 0:
        if 'source' in s23_df.columns:
            source_map = s23_df.set_index('entity_id')['source'].to_dict()
            merged['source'] = merged['s2s3_id'].map(source_map)
        else:
            merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
            
        merged['country'] = s1_df['country'].iloc[0] if len(s1_df) > 0 else "Unknown"
        merged['semantic_score'] = 0.55
        
    print(f"    [Numeric] Found {len(merged):,} candidates in {time.time()-t0:.1f}s")
    if len(merged) == 0: return pd.DataFrame()
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]
