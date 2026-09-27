import pandas as pd
import torch
import joblib, os
from sentence_transformers import SentenceTransformer

SHARED_CACHE = r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache'
data = joblib.load(os.path.join(SHARED_CACHE, 'train_data_1.pkl'))
s1 = data['val_s1'].set_index('entity_id')
s2 = data['val_s2'].set_index('entity_id')
s3 = data['val_s3'].set_index('entity_id')
s23 = pd.concat([s2, s3])

df = pd.read_csv(os.path.join(SHARED_CACHE, 'v5_missed_pairs_sample.tsv'), sep='\t').head(1000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

s1_texts = df['s1_id'].map(s1['name_address']).fillna('').tolist()
s23_texts = df['s2s3_id'].map(s23['name_address']).fillna('').tolist()

print("Encoding...")
emb1 = model.encode(s1_texts, convert_to_tensor=True, normalize_embeddings=True)
emb23 = model.encode(s23_texts, convert_to_tensor=True, normalize_embeddings=True)

scores = (emb1 * emb23).sum(dim=1).cpu().numpy()
df['true_semantic_score'] = scores
print("True Semantic Scores of Missed Pairs:")
print(df['true_semantic_score'].describe())
print(f"Pairs below 0.55: {(df['true_semantic_score'] < 0.55).mean()*100:.2f}%")
print(f"Pairs below 0.60: {(df['true_semantic_score'] < 0.60).mean()*100:.2f}%")
print(f"Pairs below 0.70: {(df['true_semantic_score'] < 0.70).mean()*100:.2f}%")
print(f"Pairs above 0.75: {(df['true_semantic_score'] >= 0.75).mean()*100:.2f}%")
