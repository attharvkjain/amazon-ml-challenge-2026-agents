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
    
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, max_features=100000, min_df=2)
    s23_matrix = vec.fit_transform(s23_df['name_address'].fillna(''))
    s1_matrix = vec.transform(s1_df['name_address'].fillna(''))
    
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
        
        for row_idx in range(sim_scores.shape[0]):
            row_scores = sim_scores[row_idx]
            k = min(top_k, len(row_scores))
            if k == 0: continue
            
            top_k_idx = np.argpartition(row_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(row_scores[top_k_idx])][::-1]
            valid = top_k_idx[row_scores[top_k_idx] > 0.10]
            
            abs_s1_idx = i + row_idx
            all_s1_idx.extend([abs_s1_idx] * len(valid))
            all_s23_idx.extend(valid)
            all_scores.extend(row_scores[valid])
            
    df = pd.DataFrame({
        's1_id': s1_ids[all_s1_idx],
        's2s3_id': s23_ids[all_s23_idx],
        'source': s23_sources[all_s23_idx],
        'country': country,
        'semantic_score': all_scores
    })
    return df

def _generate_rare_token_candidates(s1_df, s23_df, max_freq=15):
    """O(N) Hash Blocking on rare tokens."""
    def get_tokens(df, id_col):
        texts = df['name_address'].fillna('').str.lower()
        df_tok = pd.DataFrame({id_col: df['entity_id'], 'text': texts})
        df_tok['token'] = df_tok['text'].apply(lambda x: re.findall(r'\b[a-z0-9]{3,}\b', x))
        return df_tok.explode('token')[[id_col, 'token']]
        
    s1_tok = get_tokens(s1_df, 's1_id').dropna()
    s23_tok = get_tokens(s23_df, 's2s3_id').dropna()
    
    token_counts = s1_tok['token'].value_counts()
    rare_tokens = token_counts[token_counts <= max_freq].index
    
    s1_rare = s1_tok[s1_tok['token'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['token'].isin(rare_tokens)]
    
    merged = pd.merge(s1_rare, s23_rare, on='token', how='inner')
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id'])
    
    merged['source'] = merged['s2s3_id'].apply(lambda x: 'S2' if x.startswith('S2') else 'S3')
    merged['country'] = s1_df['country'].iloc[0]
    merged['semantic_score'] = 0.60
    
    return merged[['s1_id', 's2s3_id', 'source', 'country', 'semantic_score']]

def main():
    print("Loading data...")
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
    
    # Flatten ground truth
    gt_records = []
    for s1_id, matched_ids in val_gt.items():
        for mid in matched_ids:
            gt_records.append((s1_id, mid))
    gt_df = pd.DataFrame(gt_records, columns=['s1_id', 's2s3_id'])
    print(f"Total Ground Truth pairs: {len(gt_df):,}")
    
    # Run Heuristics
    print("Running current heuristics to get V5 candidate set...")
    max_scores = semantic_cands.groupby('s1_id')['semantic_score'].max()
    orphan_ids = set(max_scores[max_scores < 0.75].index)
    missed_ids = set(s1['entity_id']) - set(semantic_cands['s1_id'])
    all_orphan_ids = orphan_ids.union(missed_ids)
    
    new_cands_list = []
    countries = s1['country'].unique()
    for c in countries:
        c_s1 = s1[s1['country'] == c].copy()
        c_s23 = s23[s23['country'] == c].copy()
        c_orphans = c_s1[c_s1['entity_id'].isin(all_orphan_ids)].copy()
        
        if len(c_orphans) > 0:
            tfidf_df = _generate_tfidf_candidates(c_orphans, c_s23, top_k=5)
            new_cands_list.append(tfidf_df)
            
        rare_df = _generate_rare_token_candidates(c_s1, c_s23, max_freq=15)
        new_cands_list.append(rare_df)
        
    all_new_cands = pd.concat(new_cands_list)
    combined_cands = pd.concat([semantic_cands, all_new_cands])
    combined_cands = combined_cands.drop_duplicates(subset=['s1_id', 's2s3_id'])
    
    # Find Missed Pairs
    print("Diagnosing Misses...")
    merged_gt = pd.merge(gt_df, combined_cands[['s1_id', 's2s3_id']], on=['s1_id', 's2s3_id'], how='left', indicator=True)
    missed_gt = merged_gt[merged_gt['_merge'] == 'left_only'].drop(columns=['_merge'])
    print(f"Total Missed Pairs: {len(missed_gt):,}")
    
    # Get Max Semantic Score for Missed S1s
    missed_s1_max_scores = max_scores.reindex(missed_gt['s1_id']).fillna(0.0).values
    missed_gt['s1_max_semantic_score'] = missed_s1_max_scores
    
    # Fetch original text
    s1_dict = s1.set_index('entity_id')['name_address'].to_dict()
    s23_dict = s23.set_index('entity_id')['name_address'].to_dict()
    
    missed_gt['s1_text'] = missed_gt['s1_id'].map(s1_dict)
    missed_gt['s2s3_text'] = missed_gt['s2s3_id'].map(s23_dict)
    
    # Save a sample to file
    out_path = os.path.join(SHARED_CACHE, 'v5_missed_pairs_sample.tsv')
    missed_gt.to_csv(out_path, sep='\t', index=False)
    print(f"Saved all missed pairs with raw text to {out_path}")
    
    # Display statistics
    print("\nMissed S1 Max Semantic Score Distribution:")
    print(missed_gt['s1_max_semantic_score'].describe())
    
    print("\nTop 5 Missed Pairs (Text):")
    for idx, row in missed_gt.head(5).iterrows():
        print("-" * 50)
        print(f"S1 ({row['s1_max_semantic_score']:.2f}): {row['s1_text']}")
        print(f"S23: {row['s2s3_text']}")

if __name__ == '__main__':
    main()
