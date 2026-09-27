import pandas as pd
import joblib, os, time
import re
import jellyfish

SHARED_CACHE = r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache'
data = joblib.load(os.path.join(SHARED_CACHE, 'train_data_1.pkl'))
s1 = data['val_s1']
s23 = pd.concat([data['val_s2'], data['val_s3']])
s23['source'] = s23['entity_id'].apply(lambda x: 'S2' if str(x).startswith('S2') else 'S3')

val_pairs_path = os.path.join(SHARED_CACHE, 'val_pairs_1.pkl')
semantic_cands = joblib.load(val_pairs_path)[0]
s23_max = semantic_cands.groupby('s2s3_id')['semantic_score'].max()
orphan_s23_ids = set(s23_max[s23_max < 0.85].index).union(set(s23['entity_id']) - set(semantic_cands['s2s3_id']))

c = 'India'
c_s1 = s1[s1['country'] == c].copy()
c_s23 = s23[s23['country'] == c].copy()
c_s23_orphans = c_s23[c_s23['entity_id'].isin(orphan_s23_ids)].copy()

def _generate_phonetic_candidates(s1_df, s23_df, max_freq=50):
    t0 = time.time()
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
    
    token_counts = s1_tok['phone'].value_counts()
    rare_tokens = token_counts[token_counts <= max_freq].index
    
    s1_rare = s1_tok[s1_tok['phone'].isin(rare_tokens)]
    s23_rare = s23_tok[s23_tok['phone'].isin(rare_tokens)]
    
    merged = pd.merge(s1_rare, s23_rare, on='phone', how='inner')
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id']).copy()
    print(f"Phonetic Max Freq {max_freq}: {len(merged):,} candidates in {time.time()-t0:.1f}s")
    return merged

print("Running phonetic blocking on S23 Orphans:")
_generate_phonetic_candidates(c_s1, c_s23_orphans, max_freq=50)
