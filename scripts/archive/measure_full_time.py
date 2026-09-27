import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from config import SAMPLE_FRAC
from main import run_train

print(f"Running Full Pipeline Time Check on {SAMPLE_FRAC} split (Major Evaluation)...")
t0 = time.time()
run_train(skip_inference=False)
total_time = time.time() - t0

print(f"\nEmpirical Full Pipeline Time for 80% slice: {total_time:.2f} seconds")

time_for_100_percent = total_time * 1.25
time_with_buffer = time_for_100_percent * 1.75
print(f"Extrapolated Full Time for 100%: {time_for_100_percent/60:.2f} mins")
print(f"Full Time with 1.75x Buffer: {time_with_buffer/60:.2f} mins")
