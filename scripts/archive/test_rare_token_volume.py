import pandas as pd
import joblib, os, time
import re

SHARED_CACHE = r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache'
data = joblib.load(os.path.join(SHARED_CACHE, 'train_data_1.pkl'))
s1 = data['val_s1']
s23 = pd.concat([data['val_s2'], data['val_s3']])
s23['source'] = 'S2'

c = 'India'
c_s1 = s1[s1['country'] == c].copy()
c_s23 = s23[s23['country'] == c].copy()

def _generate_rare_token_candidates(s1_df, s23_df, max_freq=15):
    t0 = time.time()
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
    merged = merged.drop_duplicates(subset=['s1_id', 's2s3_id']).copy()
    print(f"Max Freq {max_freq}: {len(merged):,} candidates in {time.time()-t0:.1f}s")

_generate_rare_token_candidates(c_s1, c_s23, max_freq=15)
_generate_rare_token_candidates(c_s1, c_s23, max_freq=50)
_generate_rare_token_candidates(c_s1, c_s23, max_freq=100)
_generate_rare_token_candidates(c_s1, c_s23, max_freq=200)
