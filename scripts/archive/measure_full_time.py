import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC, PIPELINE_VERSION

print(f"Running Full Pipeline Time Check on {SAMPLE_FRAC} split (Major Evaluation)...")
t0 = time.time()

sample_key = f"{SAMPLE_FRAC:.1f}"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'SUBMISSION', 'code', 'business_entity_resolution', 'cache')
model_cache = os.path.join(CACHE_DIR, f'model_cache_{PIPELINE_VERSION}_{sample_key}.pkl')

if os.path.exists(model_cache):
    from main import run_test_inference
    print("Model already fully trained! Directly invoking Test Inference module...")
    run_test_inference()
else:
    from main import run_train
    run_train(skip_inference=False)

total_time = time.time() - t0
print(f"\nEmpirical Full Pipeline Time for 80% slice: {total_time:.2f} seconds")
