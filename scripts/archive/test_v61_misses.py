import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC
from preprocessing.load import load_train_data
from preprocessing.clean import preprocess_dataframe
from preprocessing.transliterate import apply_transliteration
from blocking.blocker import generate_candidates

CACHE_DIR = os.path.join(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"
data_path = os.path.join(CACHE_DIR, f'train_data_{SAMPLE_KEY}.pkl')

data = joblib.load(data_path)

print("Generating V6.1 candidates without loading from 1.0 cache...")
cands = generate_candidates(data['val_s1'], data['val_s2'], data['val_s3'], cache_prefix=f'val_k25_gnum_{SAMPLE_KEY}')

val_gt = data['val_gt']
gt_records = []
for s1_id, matched_ids in val_gt.items():
    for mid in matched_ids:
        gt_records.append((s1_id, mid))
        
gt_pairs = set(gt_records)
cand_pairs = set(tuple(x) for x in cands[['s1_id', 's2s3_id']].values)

missed = gt_pairs - cand_pairs
print(f"\nMissed Pairs: {len(missed)}")

if len(missed) > 0:
    missed_df = pd.DataFrame(list(missed), columns=['s1_id', 's2s3_id'])
    
    s1_map = data['val_s1'].set_index('entity_id')['name_address'].to_dict()
    s2_map = data['val_s2'].set_index('entity_id')['name_address'].to_dict()
    s3_map = data['val_s3'].set_index('entity_id')['name_address'].to_dict()
    
    missed_df['s1_text'] = missed_df['s1_id'].map(s1_map)
    missed_df['s2s3_text'] = missed_df['s2s3_id'].apply(lambda x: s2_map.get(x, s3_map.get(x)))
    
    missed_df.to_csv(os.path.join(CACHE_DIR, f'v61_missed_{SAMPLE_KEY}.tsv'), sep='\t', index=False)
    print(f"Saved to v61_missed_{SAMPLE_KEY}.tsv")
    print(missed_df.head(15))
