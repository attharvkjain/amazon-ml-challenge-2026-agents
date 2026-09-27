import re

with open('SUBMISSION/code/business_entity_resolution/src/main.py', encoding='utf-8') as f:
    lines = f.read()

part1 = lines.split('    # ── 8. Free memory before test data ────────────────────────────────────')[0]
part2 = lines.split('    # ── 8. Free memory before test data ────────────────────────────────────')[1]

run_train_end = '''    # ── 8. Free memory before test data ────────────────────────────────────
    print("\\n[memory] Freeing training data from memory...")
    if 'train_pairs' in locals(): del train_pairs
    if 'val_pairs' in locals(): del val_pairs
    if 'X_train' in locals(): del X_train
    if 'X_val' in locals(): del X_val
    if 'data' in locals():
        for key in list(data.keys()):
            del data[key]
        del data
    import gc; gc.collect()

    if not skip_inference:
        run_test_inference(matcher, best_threshold)
        
    return {'val_f05': best_f05}

def run_test_inference(matcher=None, best_threshold=None):
    """Standalone module to run test inference without loading training memory."""
    import time, os, gc, joblib
    from config import CACHE_DIR, PIPELINE_VERSION, SAMPLE_FRAC, OUTPUT_DIR
    from blocking.blocker import generate_candidates
    from features.similarity import extract_features
    from postprocessing.format import format_and_save_output
    from evaluation.metrics import run_validation as _run_validation
    from preprocessing.load import load_test_data
    from preprocessing.clean import _preprocess_all
    
    sample_key = f"{SAMPLE_FRAC:.1f}"
    
    # If resuming a crashed pipeline, we load the cached models
    if matcher is None or best_threshold is None:
        model_cache_path = os.path.join(CACHE_DIR, f'model_cache_{PIPELINE_VERSION}_{sample_key}.pkl')
        model_data = joblib.load(model_cache_path)
        matcher = model_data['matcher']
        best_threshold = model_data['best_threshold']
        
    timing_dict = {}

'''

part2_body = part2.split('    if skip_inference:\n        print("\\n" + "="*60)\n        print("STAGE 8-12: SKIPPING TEST INFERENCE AS REQUESTED")\n        print("="*60)\n        return\n\n')[1]

part2_body = part2_body.replace('    total_time = time.time() - total_start\n', '')
part2_body = part2_body.replace('    print(f"PIPELINE COMPLETE in {total_time/60:.1f} minutes")\n', '    print("PIPELINE COMPLETE")\n')

part2_body = part2_body.split('    return {\n')[0]

if 'def run_cv():' in lines:
    cv_part = 'def run_cv():' + lines.split('def run_cv():')[1]
    new_file = part1 + run_train_end + part2_body + '\n\n' + cv_part
else:
    new_file = part1 + run_train_end + part2_body

with open('SUBMISSION/code/business_entity_resolution/src/main.py', 'w', encoding='utf-8') as f:
    f.write(new_file)
