import time
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC
from preprocessing.load import load_train_data
from preprocessing.clean import preprocess_dataframe
from preprocessing.transliterate import apply_transliteration
from blocking.blocker import generate_candidates

CACHE_DIR = os.path.join(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache')
SAMPLE_KEY = f"{SAMPLE_FRAC:g}"
data_path = os.path.join(CACHE_DIR, f'train_data_{SAMPLE_KEY}.pkl')

if not os.path.exists(data_path):
    print("Loading data...")
    data = load_train_data(sample_frac=SAMPLE_FRAC, val_frac=0.2)
    for key in data:
        if 'gt' not in key:
            data[key] = preprocess_dataframe(data[key])
            data[key] = apply_transliteration(data[key], is_s1=('s1' in key))
    joblib.dump(data, data_path)
else:
    data = joblib.load(data_path)

print(f"Running Empirical Time Check on {SAMPLE_KEY} split...")
t0 = time.time()
generate_candidates(data['val_s1'], data['val_s2'], data['val_s3'], cache_prefix='val_empirical')
total_time = time.time() - t0

print(f"\nEmpirical Time for 7.5% slice: {total_time:.2f} seconds")

time_for_100_percent = total_time * 10
time_with_buffer = time_for_100_percent * 1.75
print(f"Extrapolated Time for 100%: {time_for_100_percent/60:.2f} mins")
print(f"Time with 1.75x Buffer: {time_with_buffer/60:.2f} mins")
