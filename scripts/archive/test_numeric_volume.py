import sys
import os
import joblib
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC
from blocking.heuristic_blocker import _generate_rare_token_candidates

CACHE_DIR = os.path.join(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"
data = joblib.load(os.path.join(CACHE_DIR, f'train_data_{SAMPLE_KEY}.pkl'))

s1_c = data['val_s1']
s2_c = data['val_s2']
s3_c = data['val_s3']
s23_c = pd.concat([s2_c, s3_c])

# Extract numeric tokens
def get_numeric_tokens_df(df, text_col='name_address', id_col='entity_id'):
    records = []
    for _, row in df.iterrows():
        text = str(row[text_col]).lower()
        tokens = [t for t in text.split() if any(c.isdigit() for c in t)]
        for t in set(tokens):
            records.append((row[id_col], t))
    return pd.DataFrame(records, columns=[id_col, 'token'])

t0 = time.time()
s1_tokens = get_numeric_tokens_df(s1_c)
s23_tokens = get_numeric_tokens_df(s23_c)

# Calculate frequencies in S23
token_counts = s23_tokens['token'].value_counts()
rare_tokens = token_counts[token_counts <= 500].index

s23_rare = s23_tokens[s23_tokens['token'].isin(rare_tokens)]
s1_rare = s1_tokens[s1_tokens['token'].isin(rare_tokens)]

merged = pd.merge(s1_rare, s23_rare, on='token')
cands = merged[['entity_id_x', 'entity_id_y']].drop_duplicates()
cands.columns = ['s1_id', 's2s3_id']

print(f"Numeric Hash (freq <= 500) generated {len(cands):,} candidates in {time.time()-t0:.2f}s")
