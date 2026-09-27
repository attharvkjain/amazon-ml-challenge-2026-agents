import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC
from blocking.blocker import compute_blocking_recall
from blocking.heuristic_blocker import _generate_rare_token_candidates, _generate_phonetic_candidates

SHARED_CACHE = os.path.join(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"

data = joblib.load(os.path.join(SHARED_CACHE, f'train_data_{SAMPLE_KEY}.pkl'))
s1 = data['val_s1']
s23 = pd.concat([data['val_s2'], data['val_s3']])
s23['source'] = 'S2'
val_gt = data['val_gt']

semantic_cands = joblib.load(os.path.join(SHARED_CACHE, f'val_pairs_{SAMPLE_KEY}.pkl'))[0]

s23_max = semantic_cands.groupby('s2s3_id')['semantic_score'].max()
orphan_s23_ids = set(s23_max[s23_max < 0.85].index).union(set(s23['entity_id']) - set(semantic_cands['s2s3_id']))

new_cands_list = []
countries = s1['country'].unique()
for c in countries:
    c_s1 = s1[s1['country'] == c]
    c_s23 = s23[s23['country'] == c]
    c_s23_orphans = c_s23[c_s23['entity_id'].isin(orphan_s23_ids)]
    
    new_cands_list.append(_generate_rare_token_candidates(c_s1, c_s23_orphans, max_freq=500))
    new_cands_list.append(_generate_phonetic_candidates(c_s1, c_s23_orphans, max_freq=200))

combined = pd.concat([semantic_cands] + new_cands_list).drop_duplicates(subset=['s1_id', 's2s3_id'])
compute_blocking_recall(combined, val_gt)
