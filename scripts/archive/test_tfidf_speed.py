import time
import joblib, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SHARED_CACHE = r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache'
data = joblib.load(os.path.join(SHARED_CACHE, 'train_data_1.pkl'))
s1 = data['val_s1'].head(1000000)
s2 = data['val_s2'].head(50000)

print("Fitting TF-IDF...")
t0 = time.time()
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), max_features=100000, min_df=2)
s1_matrix = vec.fit_transform(s1['name_address'].fillna(''))
s2_matrix = vec.transform(s2['name_address'].fillna(''))
print(f"Vectorized in {time.time()-t0:.1f}s")

s1_matrixT = s1_matrix.T.tocsr()

print("Dot product...")
t1 = time.time()
chunk = s2_matrix[:1000]
sim = chunk.dot(s1_matrixT).toarray()
print(f"1000 queries vs 1M docs took {time.time()-t1:.3f}s")
