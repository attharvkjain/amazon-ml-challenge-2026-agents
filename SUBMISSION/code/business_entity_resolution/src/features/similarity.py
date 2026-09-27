"""
Pairwise feature extraction for candidate pairs.
(Multithreading optimized)
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import multiprocessing
from joblib import Parallel, delayed

FEATURE_NAMES = [
    "jaro_winkler_name", "levenshtein_ratio_name",
    "token_sort_ratio_name", "token_set_ratio_name",
    "jaro_winkler_address", "levenshtein_ratio_address",
    "token_sort_ratio_address", "token_set_ratio_address",
    "token_overlap_name", "token_overlap_address",
    "name_length_ratio", "address_length_ratio",
    "numbers_jaccard_name", "numbers_jaccard_address",
    "source_indicator",
    "semantic_score",
]

_NUM_RE = re.compile(r'\d+')


def _token_overlap_jaccard(s1: str, s2: str) -> float:
    tokens1 = set(s1.split()) if s1 else set()
    tokens2 = set(s2.split()) if s2 else set()
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def _length_ratio(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)

def _numbers_jaccard(s1: str, s2: str) -> float:
    nums1 = set(_NUM_RE.findall(s1)) if s1 else set()
    nums2 = set(_NUM_RE.findall(s2)) if s2 else set()
    if not nums1 and not nums2:
        return 1.0  # Both have no numbers -> perfectly consistent
    if not nums1 or not nums2:
        return 0.0  # One has numbers, one doesn't -> inconsistent
    return len(nums1 & nums2) / len(nums1 | nums2)


def compute_pair_features(name1: str, name2: str, addr1: str, addr2: str, source: str, semantic_score: float) -> list[float]:
    name1 = name1 or ""
    name2 = name2 or ""
    addr1 = addr1 or ""
    addr2 = addr2 or ""
    return [
        JaroWinkler.normalized_similarity(name1, name2),
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.token_set_ratio(name1, name2) / 100.0,
        JaroWinkler.normalized_similarity(addr1, addr2),
        fuzz.ratio(addr1, addr2) / 100.0,
        fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        fuzz.token_set_ratio(addr1, addr2) / 100.0,
        _token_overlap_jaccard(name1, name2),
        _token_overlap_jaccard(addr1, addr2),
        _length_ratio(name1, name2),
        _length_ratio(addr1, addr2),
        _numbers_jaccard(name1, name2),
        _numbers_jaccard(addr1, addr2),
        1.0 if source == "S3" else 0.0,
        float(semantic_score),
    ]


def _extract_chunk_arrays(name1_arr, addr1_arr, name2_arr, addr2_arr, source_arr, score_arr) -> np.ndarray:
    """Process a chunk of pre-mapped string arrays. Executed in worker process (loky)."""
    features = []
    for n1, a1, n2, a2, src, sc in zip(name1_arr, addr1_arr, name2_arr, addr2_arr, source_arr, score_arr):
        feat = compute_pair_features(n1, n2, a1, a2, src, sc)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def extract_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> np.ndarray:
    """Extract pairwise features avoiding GIL by running Python loops in loky workers."""
    if len(pairs_df) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    n_jobs = max(1, multiprocessing.cpu_count() - 2)

    print(f"[features] Preparing {len(pairs_df):,} pairs for multiprocessing...")
    
    # 1. Build rapid lookups in the main thread (Pandas Series)
    s1_names = s1_df.set_index('entity_id')['clean_name']
    s1_addrs = s1_df.set_index('entity_id')['clean_address']
    
    s2s3_df = pd.concat([s2_df, s3_df])
    s2s3_names = s2s3_df.set_index('entity_id')['clean_name']
    s2s3_addrs = s2s3_df.set_index('entity_id')['clean_address']

    # 2. Split into MICRO-CHUNKS to prevent Loky Pickling OOM
    # 14 chunks in memory = ~5.6M strings = ~300 MB payload
    chunk_size = 50_000
    chunks = [pairs_df.iloc[i:i + chunk_size] for i in range(0, len(pairs_df), chunk_size)]
    
    def prepare_chunk(chunk_df):
        # Map IDs to strings in main thread
        n1 = chunk_df['s1_id'].map(s1_names).fillna("").values
        a1 = chunk_df['s1_id'].map(s1_addrs).fillna("").values
        n2 = chunk_df['s2s3_id'].map(s2s3_names).fillna("").values
        a2 = chunk_df['s2s3_id'].map(s2s3_addrs).fillna("").values
        src = chunk_df['source'].values
        sc = chunk_df['semantic_score'].values if 'semantic_score' in chunk_df.columns else np.zeros(len(chunk_df))
        return (n1, a1, n2, a2, src, sc)

    print(f"[features] Extracting features on {n_jobs} cores in batches to bound memory overhead...")
    
    # Pre-allocate to prevent np.vstack OOM on 13GB arrays
    results_matrix = np.empty((len(pairs_df), len(FEATURE_NAMES)), dtype=np.float32)
    current_idx = 0
    
    # Process batch by batch so we don't hold 48GB of strings in memory simultaneously
    batch_size = n_jobs
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_data = [prepare_chunk(c) for c in batch_chunks]
        
        batch_res = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_extract_chunk_arrays)(*data) for data in batch_data
        )
        
        for chunk_arr in batch_res:
            chunk_len = len(chunk_arr)
            results_matrix[current_idx:current_idx+chunk_len] = chunk_arr
            current_idx += chunk_len
        
        # Explicit cleanup to ensure garbage collection frees the string arrays
        del batch_data, batch_res
        import gc; gc.collect()

    # Free mapping series
    del s1_names, s1_addrs, s2s3_names, s2s3_addrs, s2s3_df
    
    return results_matrix


def generate_labels(pairs_df: pd.DataFrame, ground_truth: dict[str, set[str]]) -> np.ndarray:
    labels = []
    for row in pairs_df.itertuples(index=False):
        gt_set = ground_truth.get(row.s1_id, set())
        labels.append(1 if row.s2s3_id in gt_set else 0)

    labels = np.array(labels, dtype=np.int32)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos > 0:
        print(f"[features] Labels: {pos:,} pos, {neg:,} neg (ratio 1:{neg/pos:.1f})")
    return labels
