"""
Blocking - country-first hard partition plus GPU Semantic Blocking candidate generation,
augmented with Single-Pass Conditional Heuristics (TF-IDF & Rare Tokens).
"""
from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import multiprocessing

import torch
import joblib
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLOCKING_TOP_K, OUTPUT_DIR

# Import heuristic blockers
from blocking.heuristic_blocker import _generate_tfidf_candidates, _generate_rare_token_candidates, _generate_phonetic_candidates, _generate_bigram_candidates, _generate_numeric_candidates

def _dense_top_k(query_embeddings: torch.Tensor, index_embeddings: torch.Tensor, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched dense matrix top-K search on GPU.
    Uses chunked exact cosine similarity (dot product of normalized vectors).
    Returns fully vectorized arrays: (q_idx, s1_idx, scores)
    """
    chunk_size = 256  # Small chunk size for 8GB VRAM with 700k records
    n_queries = query_embeddings.shape[0]
    
    k_actual = min(top_k, index_embeddings.shape[0])
    total_elements = n_queries * k_actual
    
    # Pre-allocate output arrays to entirely prevent list-accumulation OOM/thrashing
    q_idx_arr = np.empty(total_elements, dtype=np.int32)
    s1_idx_arr = np.empty(total_elements, dtype=np.int32)
    scores_arr = np.empty(total_elements, dtype=np.float32)
    
    current_idx = 0
    
    # Process queries in chunks
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk = query_embeddings[start:end].to(index_embeddings.device)
        
        # Exact cosine similarity (assuming normalized vectors)
        sim_matrix = torch.matmul(chunk, index_embeddings.T)
        
        # Get Top-K
        top_scores, top_indices = torch.topk(sim_matrix, k_actual, dim=1)
        
        # Move to CPU to free VRAM for next operations
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Vectorized array construction
        chunk_len = end - start
        elements_in_chunk = chunk_len * k_actual
        q_idxs = np.arange(start, end).reshape(-1, 1).repeat(k_actual, axis=1)
        
        # Direct memory assignment
        q_idx_arr[current_idx : current_idx + elements_in_chunk] = q_idxs.flatten()
        s1_idx_arr[current_idx : current_idx + elements_in_chunk] = top_indices.flatten()
        scores_arr[current_idx : current_idx + elements_in_chunk] = top_scores.flatten()
        
        current_idx += elements_in_chunk
            
    return q_idx_arr, s1_idx_arr, scores_arr


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int | None = None,
    save_path: str | os.PathLike | None = None,
    target_country: str | None = None,
    cache_prefix: str = 'train',
) -> pd.DataFrame:
    """Generate candidate pairs using Semantic GPU blocking + CPU Heuristics."""
    if top_k is None:
        top_k = BLOCKING_TOP_K

    if target_country:
        countries = [target_country]
    else:
        countries = sorted(s1['country'].unique())
        
    print(f"[blocker] Countries: {countries}")
    all_pairs_dfs = []
    
    # ── Inter-stage Caching ──
    blocker_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache')
    os.makedirs(blocker_cache_dir, exist_ok=True)

    print("  Loading MiniLM-L12-v2 to GPU ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    for country in countries:
        print(f"\n[blocker] Processing country: {country}")
        country_cache_path = os.path.join(blocker_cache_dir, f'blocker_cache_{cache_prefix}_{country}.pkl')
        if os.path.exists(country_cache_path):
            print(f"  [cache] Loading {country} candidates from cache...")
            cached_dfs = joblib.load(country_cache_path)
            all_pairs_dfs.extend(cached_dfs)
            continue
            
        s1_c = s1[s1['country'] == country].reset_index(drop=True)
        s2_c = s2[s2['country'] == country].reset_index(drop=True)
        s3_c = s3[s3['country'] == country].reset_index(drop=True)

        print(f"  S1: {len(s1_c):,}, S2: {len(s2_c):,}, S3: {len(s3_c):,}")

        if len(s1_c) == 0:
            continue
        
        semantic_dfs_for_country = []
        
        s2_df_cache_path = os.path.join(blocker_cache_dir, f'blocker_cache_{cache_prefix}_{country}_s2_df.pkl')
        s3_df_cache_path = os.path.join(blocker_cache_dir, f'blocker_cache_{cache_prefix}_{country}_s3_df.pkl')

        need_s1_encode = False
        if len(s2_c) > 0 and not os.path.exists(s2_df_cache_path):
            need_s1_encode = True
        if len(s3_c) > 0 and not os.path.exists(s3_df_cache_path):
            need_s1_encode = True

        if need_s1_encode:
            print(f"  Encoding S1 ...")
            s1_texts = s1_c['name_address'].fillna('').tolist()
            s1_embeddings = model.encode(s1_texts, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, device=device, show_progress_bar=True)
            del s1_texts
            gc.collect()
            
        s1_ids = s1_c['entity_id'].values

        if len(s2_c) > 0:
            if os.path.exists(s2_df_cache_path):
                print(f"  [cache] Loading S2 from cache...")
                s2_df = joblib.load(s2_df_cache_path)
                semantic_dfs_for_country.append(s2_df)
            else:
                print(f"  Encoding and Blocking S2 ({len(s2_c):,} records) ...")
                s2_texts = s2_c['name_address'].fillna('').tolist()
                s2_embeddings = model.encode(s2_texts, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, device=device, show_progress_bar=True)
                s2_embeddings = s2_embeddings.cpu()
                del s2_texts  # Free ~500MB of string list before top_k
                if device == 'cuda': torch.cuda.empty_cache()
                gc.collect()
                
                s2_results = _dense_top_k(s2_embeddings, s1_embeddings, top_k)
                s2_ids = s2_c['entity_id'].values

                q_idx_arr, s1_idx_arr, scores_arr = s2_results
                del s2_embeddings, s2_results
                if device == 'cuda': torch.cuda.empty_cache()
                gc.collect()
                
                s2_df = pd.DataFrame({
                    's1_id': s1_ids[s1_idx_arr],
                    's2s3_id': s2_ids[q_idx_arr],
                    'source': pd.Categorical(['S2'] * len(s1_idx_arr)),
                    'country': pd.Categorical([country] * len(s1_idx_arr)),
                    'semantic_score': scores_arr
                })
                del q_idx_arr, s1_idx_arr, scores_arr
                s2_df = s2_df[s2_df['semantic_score'] >= 0.55].reset_index(drop=True)
                joblib.dump(s2_df, s2_df_cache_path)
                semantic_dfs_for_country.append(s2_df)
                gc.collect()

        if len(s3_c) > 0:
            if os.path.exists(s3_df_cache_path):
                print(f"  [cache] Loading S3 from cache...")
                s3_df = joblib.load(s3_df_cache_path)
                semantic_dfs_for_country.append(s3_df)
            else:
                print(f"  Encoding and Blocking S3 ({len(s3_c):,} records) ...")
                s3_texts = s3_c['name_address'].fillna('').tolist()
                s3_embeddings = model.encode(s3_texts, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, device=device, show_progress_bar=True)
                s3_embeddings = s3_embeddings.cpu()
                del s3_texts
                if device == 'cuda': torch.cuda.empty_cache()
                gc.collect()
                
                s3_results = _dense_top_k(s3_embeddings, s1_embeddings, top_k)
                s3_ids = s3_c['entity_id'].values

                q_idx_arr, s1_idx_arr, scores_arr = s3_results
                del s3_embeddings, s3_results
                if device == 'cuda': torch.cuda.empty_cache()
                gc.collect()
                
                s3_df = pd.DataFrame({
                    's1_id': s1_ids[s1_idx_arr],
                    's2s3_id': s3_ids[q_idx_arr],
                    'source': pd.Categorical(['S3'] * len(s1_idx_arr)),
                    'country': pd.Categorical([country] * len(s1_idx_arr)),
                    'semantic_score': scores_arr
                })
                del q_idx_arr, s1_idx_arr, scores_arr
                s3_df = s3_df[s3_df['semantic_score'] >= 0.55].reset_index(drop=True)
                joblib.dump(s3_df, s3_df_cache_path)
                semantic_dfs_for_country.append(s3_df)
                gc.collect()

        if need_s1_encode:
            del s1_embeddings
            if device == 'cuda': torch.cuda.empty_cache()
            gc.collect()
        
        # ── V6 Heuristic Blocking ──
        print("  Applying V6 Single-Pass Heuristics...", flush=True)
        c_semantic = pd.concat(semantic_dfs_for_country, ignore_index=True) if semantic_dfs_for_country else pd.DataFrame()
        del semantic_dfs_for_country
        gc.collect()
        
        s23_c = pd.concat([s2_c, s3_c])
        if 'source' not in s23_c.columns:
            s23_c['source'] = s23_c['entity_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')
            
        streams = [c_semantic]
        
        if len(c_semantic) > 0 and len(s1_c) > 0:
            # 1. S1 Orphan TF-IDF (Score < 0.75)
            # Use sort_values + drop_duplicates instead of groupby to avoid 30GB pandas string hash-table OOM on 100M+ rows
            max_scores = c_semantic[['s1_id', 'semantic_score']].sort_values('semantic_score', ascending=False).drop_duplicates('s1_id').set_index('s1_id')['semantic_score']
            orphan_ids = set(max_scores[max_scores < 0.75].index)
            missing_ids = set(s1_c['entity_id']) - set(c_semantic['s1_id'])
            all_orphan_ids = orphan_ids.union(missing_ids)
            
            print(f"    -> Identified {len(all_orphan_ids):,} S1 orphans for targeted TF-IDF", flush=True)
            orphan_s1_c = s1_c[s1_c['entity_id'].isin(all_orphan_ids)].copy()
            
            # 2. S23 Orphan Detection (Score < 0.85)
            s23_max_scores = c_semantic[['s2s3_id', 'semantic_score']].sort_values('semantic_score', ascending=False).drop_duplicates('s2s3_id').set_index('s2s3_id')['semantic_score']
            orphan_s23_ids = set(s23_max_scores[s23_max_scores < 0.85].index)
            missing_s23 = set(s23_c['entity_id']) - set(c_semantic['s2s3_id'])
            all_orphan_s23_ids = orphan_s23_ids.union(missing_s23)
            orphan_s23_c = s23_c[s23_c['entity_id'].isin(all_orphan_s23_ids)].copy()
            print(f"    -> Identified {len(all_orphan_s23_ids):,} S2/S3 orphans for expanded hash blocking", flush=True)
            
            # 3. Parallelize Heuristics Generation
            from concurrent.futures import ThreadPoolExecutor
            
            def run_tfidf():
                if len(orphan_s1_c) > 0:
                    return _generate_tfidf_candidates(orphan_s1_c, s23_c, top_k=5)
                return pd.DataFrame()
                
            def run_rare():
                return _generate_rare_token_candidates(s1_c, s23_c, max_freq=15)
                
            def run_bigram():
                if len(orphan_s23_c) > 0:
                    return _generate_bigram_candidates(s1_c, orphan_s23_c, max_freq=100)
                return pd.DataFrame()
                
            def run_phonetic():
                if len(orphan_s23_c) > 0:
                    return _generate_phonetic_candidates(s1_c, orphan_s23_c, max_freq=50)
                return pd.DataFrame()
                
            def run_numeric():
                return _generate_numeric_candidates(s1_c, s23_c, max_freq=500)
                
            print("    -> Executing 5 heuristic algorithms in parallel...", flush=True)
            with ThreadPoolExecutor(max_workers=5) as executor:
                f_tfidf = executor.submit(run_tfidf)
                f_rare = executor.submit(run_rare)
                f_bigram = executor.submit(run_bigram)
                f_phonetic = executor.submit(run_phonetic)
                f_numeric = executor.submit(run_numeric)
                
                streams.extend([f_tfidf.result(), f_rare.result(), f_bigram.result(), f_phonetic.result(), f_numeric.result()])
            
        print("  Unioning all streams...", flush=True)
        combined_df = pd.concat(streams, ignore_index=True)
        del streams
        gc.collect()
        
        if len(combined_df) > 0:
            # Memory Optimization: Drop duplicates via integer bitwise hashing 
            # Chunked to prevent string-slice allocation OOM on 150M+ rows
            print("    [De-dup] Computing hashes in chunks...", flush=True)
            chunk_size = 10_000_000
            hashes = np.empty(len(combined_df), dtype=np.uint64)
            
            is_s3_global = (combined_df['source'] == 'S3').values
            
            for start in range(0, len(combined_df), chunk_size):
                end = min(start + chunk_size, len(combined_df))
                
                # Fast chunked string slice and cast
                s1_chunk = combined_df['s1_id'].iloc[start:end].astype(str).str.slice(3).astype(np.uint64)
                s2_chunk = combined_df['s2s3_id'].iloc[start:end].astype(str).str.slice(3).astype(np.uint64)
                
                is_s3_chunk = is_s3_global[start:end]
                s2_chunk.values[is_s3_chunk] += 2_000_000_000  # Offset S3 to avoid S2 collision
                
                hashes[start:end] = (s1_chunk.values << 32) | s2_chunk.values
                
            print("    [De-dup] Extracting unique indices...", flush=True)
            _, unique_indices = np.unique(hashes, return_index=True)
            
            # FREE MEMORY immediately
            del hashes
            gc.collect()
            
            # Sort indices to preserve memory locality and prevent massive scattered copy overhead
            unique_indices.sort()
            
            combined_df = combined_df.take(unique_indices)
            combined_df.reset_index(drop=True, inplace=True)
            
        country_dfs = [combined_df]

        print(f"  [cache] Saving {country} candidates to cache ({len(combined_df):,} total pairs)...")
        joblib.dump(country_dfs, country_cache_path)
        all_pairs_dfs.extend(country_dfs)

    pairs_df = pd.concat(all_pairs_dfs, ignore_index=True) if all_pairs_dfs else pd.DataFrame()
    print(f"\n[blocker] Total candidate pairs: {len(pairs_df):,}")

    if save_path is not None:
        _save_candidate_pairs(pairs_df, s1, save_path)

    return pairs_df


def _save_candidate_pairs(pairs_df: pd.DataFrame, s1: pd.DataFrame, save_path: str | os.PathLike) -> None:
    import collections
    cand_dict = collections.defaultdict(set)
    for s1_id, s2_id in zip(pairs_df['s1_id'].values, pairs_df['s2s3_id'].values):
        cand_dict[s1_id].add(s2_id)
        
    grouped_s1, grouped_s2 = [], []
    for k, v in cand_dict.items():
        grouped_s1.append(k)
        grouped_s2.append(','.join(sorted(v)))
        
    grouped = pd.DataFrame({'source1_entity_id': grouped_s1, 'candidate_entity_ids': grouped_s2})
    all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id'].unique()})
    result = all_s1.merge(grouped, on='source1_entity_id', how='left')
    result['candidate_entity_ids'] = result['candidate_entity_ids'].fillna('')
    result.to_csv(save_path, sep='\t', index=False)
    print(f"[blocker] Saved candidate_pairs.tsv: {len(result):,} rows -> {save_path}")


def compute_blocking_recall(pairs_df: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
    # 1. Flatten ground truth into a DataFrame (extremely fast for ~3M pairs)
    gt_records = []
    for s1_id, matched_ids in ground_truth.items():
        for mid in matched_ids:
            gt_records.append((s1_id, mid))
    
    total_true = len(gt_records)
    if total_true == 0:
        return 0.0
        
    gt_df = pd.DataFrame(gt_records, columns=['s1_id', 's2s3_id'])
    
    # 2. Chunked vectorized merge (strictly bounds memory overhead to prevent Int64Vector OOM)
    chunk_size = 5_000_000
    found = 0
    pairs_subset = pairs_df[['s1_id', 's2s3_id']]
    
    for i in range(0, len(pairs_subset), chunk_size):
        chunk = pairs_subset.iloc[i:i+chunk_size]
        merged = chunk.merge(gt_df, on=['s1_id', 's2s3_id'], how='inner')
        found += len(merged)
    
    recall = found / total_true
    print(f"[blocker] Blocking recall: {found:,}/{total_true:,} = {recall:.4f}")
    return recall
