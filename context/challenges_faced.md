> **Version:** v2.3 | **Last updated:** 2026-09-26 22:08 IST | **By:** Antigravity

# Challenges, Pitfalls, and Resolutions

This document logs all errors, crashes, performance bottlenecks, and design issues encountered during the challenge, along with their solutions. Review this to avoid repeating mistakes.

## 1. Feature Extraction Bottleneck (`iterrows` + GIL)
- **Problem:** Stage 4 (Feature Extraction) on 27.5 million candidate pairs was taking an unreasonably long time (projected >1 hour). We initially used the `threading` backend in `joblib.Parallel` and iterated through candidate chunks using `pandas.DataFrame.iterrows()`. The combination of `iterrows()` overhead and Python's Global Interpreter Lock (GIL) contention completely starved the CPU.
- **Resolution:** Switched to the `loky` backend for true separate processes (bypassing the GIL) and replaced `.iterrows()` with `.itertuples(index=False)`, resulting in a near 100x speedup.

## 2. Windows Terminal Encoding Crash (`UnicodeEncodeError`)
- **Problem:** At the end of LightGBM training (Stage 6), the pipeline crashed entirely, throwing a `UnicodeEncodeError`. The script attempted to print `F₀.₅` and `→` to the console, which Windows Command Prompt / PowerShell (`cp1252` encoding) could not handle.
- **Resolution:** Replaced all Unicode subscripts and arrows (`F₀.₅`, `→`) with standard ASCII equivalents (`F0.5`, `->`) across `main.py`, `metrics.py`, and `threshold.py`.

## 3. Lost Pipeline Progress
- **Problem:** Because of the `UnicodeEncodeError` crash above, 15 minutes of intensive feature extraction for the training set was lost and had to be recomputed from scratch, posing a major risk for the upcoming 2-hour test set extraction.
- **Resolution:** Integrated explicit pipeline caching in `main.py`. Intermediate states (cleaned data, candidate pairs, extracted feature matrices) are now saved to a `cache/` directory using `joblib.dump()`. If the pipeline crashes, it will resume from the nearest checkpoint on restart.

## 4. OOM (Out of Memory) during Test Set Blocking
- **Problem:** In `blocker.py`, blocking the massive Test Set caused severe `ArrayMemoryError` crashes. Initially, this was because the sparse matrix was being unnecessarily converted to a dense numpy array (consuming 1 GB per batch of 500). After rewriting the code to operate directly on the sparse matrix to avoid dense conversion, we still hit OOMs! The root cause is that character 3-gram TF-IDF vectors have incredibly high overlap (over 98% density on India and US). Because sparse matrices must store both data and index arrays, a 98% dense sparse matrix actually consumes *more* memory than a dense array. A batch size of 200 on India's 809k records generated 160 million non-zero edges, allocating 600MB per thread (8.5 GB total) and crashing the pipeline. If left at 200, the US dataset (3.1M records) would have allocated 24 GB and instantly crashed.
- **Resolution:** Implemented **Dynamic Batch Sizing** in `_sparse_top_k`. The `batch_size` is now calculated on the fly as `50_000_000 // n_index_records`, strictly capping the maximum possible non-zero elements per batch at 50 million (~200MB per thread, 2.8GB total across 14 threads). This guarantees bulletproof stability across all countries regardless of their size or TF-IDF density, while still fully saturating the CPU.

## 5. IPC Deserialization Bottleneck
- **Problem:** During multithreaded feature extraction (Stage 4 and Stage 10), returning raw Python lists of floats from the worker processes back to the main process created a massive Inter-Process Communication (IPC) serialization overhead. Transferring 3GB of lists took several minutes on a single CPU thread on the main process while the rest of the cores idled.
- **Resolution:** Modified `_extract_chunk` in `similarity.py` to immediately convert the extracted features into a C-level `np.ndarray` of `np.float32` *before* returning them across the IPC pipe, and used `np.vstack()` on the master thread. This allows zero-copy memory mapping, completely eliminating the IPC bottleneck.

---

## 6. Massive 30-Hour Inference Time on Test Set
- **Problem:** After solving the OOMs, the pipeline ran successfully but was projected to take ~30 hours to finish the Test Set. The bottleneck was `blocker.py`: using `analyzer='char_wb'` and `ngram_range=(3,3)` creates massive overlap between businesses. Slicing 10 million companies into character 3-grams generated a 70% dense sparse matrix, requiring over 10 Trillion mathematical dot-products for the US and India, taking 29 hours.
- **Resolution:** Pivoted the Test Set inference to use **Word Unigrams** (`analyzer='word'`, `ngram_range=(1,1)`). At the time, the run used `max_df=0.25`; the current baseline setting is `max_df=0.01` in `src/blocking/blocker.py`. Because random companies rarely share exact words (unless generic), the matrix density plummeted to `<0.1%`. This sped up the dot product by 100x, allowing inference to finish in under 30 minutes! While it slightly reduces candidate recall on severe typos, the speedup was critical for the hackathon crunch.

## 7. Pandas BlockManager 6GB RAM Spike
- **Problem:** When collecting the 94 Million India candidate pairs into a `pd.DataFrame` during blocking, the pipeline repeatedly crashed with `std::bad_alloc` `ArrayMemoryError: Unable to allocate 1.43 GiB for an array...`. Pandas' internal `BlockManager` attempts to aggressively merge contiguous string/object columns into a single 2D Numpy array block of pointers. Trying to merge `s1_id`, `s2s3_id`, `source`, and `country` created a 1.5GB pointer block requirement that Windows could not physically contiguous-allocate. Additionally, passing `np.array(s1_ids)[...]` instantiated a 6GB unicode string array in RAM before Pandas even touched it.
- **Resolution:** Forcefully circumvented the `BlockManager`. Replaced numpy slicing with standard python list comprehension (`[s1_ids[i] for i in ...]`) which just creates tiny pointers to interned strings (384MB). Then, created an empty `pd.DataFrame()` and added the columns sequentially, casting `source` and `country` as `pd.Categorical`. This mathematically prevents Pandas from attempting to allocate massive 2D pointer blocks, dropping peak RAM usage from 6GB down to <1GB for 94 Million rows.

## 8. Loky Process Pickling Limit on Massive DataFrames
- **Problem:** In `similarity.py`, feature extraction for the 94 Million India pairs crashed with `_pickle.PicklingError` and `MemoryError` in `loky`. Joblib's `loky` backend spawns separate Python processes, which forces the main process to serialize (pickle) the entire 94M row DataFrame and 5M key string lookup dictionaries into IPC pipes. The memory required to pickle 3GB of raw text crashed the system instantly.
- **Resolution:** Switched `joblib` from `backend='loky'` to `backend='threading'`. Because the bottleneck is the string edit distances calculated inside the C++ `RapidFuzz` library (which releases the Python GIL), multithreading allows 100% CPU utilization across all 14 cores while letting all threads passively share the memory of the original DataFrame without any pickling overhead whatsoever.

## 9. GitHub 100MB File Size Limit Crash
- **Problem:** When attempting to `git push` the final branch to GitHub, the remote server rejected the push (`pre-receive hook declined: GH001: Large files detected`) because a 118MB `submission_002_matching_results.tsv` file had been committed to the history by an automated reconciliation agent.
- **Resolution:** Performed a `git reset --soft` to rewind the local branch history without losing any file modifications. Manually unstaged the massive `.tsv` file, added it to `.gitignore`, and squashed all the valid changes into a single new commit, completely removing the large file from the Git tree and successfully pushing to the remote.

## 10. RTX 5000 Series (Blackwell) PyTorch CUDA Incompatibility
- **Problem:** During the V2 Semantic GPU Blocking upgrade, PyTorch fell back to CPU speeds (1s per batch of 256). Checking `torch.cuda.is_available()` revealed it was attempting to run, but threw a fatal warning: `NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible with the current PyTorch installation.` Standard PyTorch 2.5 (`cu121`) wheels predate Blackwell (`sm_120`) and strip PTX, preventing JIT compilation for newer architectures. We even tried PyTorch 2.6 (`cu124`) and 2.14 (`cu126`), which still lacked the binaries and explicitly logged that we needed CUDA 13.0+.
- **Resolution:** Upgraded to `torch==2.14.0+cu130` directly from the PyTorch `cu130` index, which finally included the Blackwell `sm_120` kernels, instantly restoring full GPU acceleration for `sentence-transformers` and allowing batch size to be increased to 1024.

## 11. Skewed Public Leaderboard & Generalization Gap
- **Problem:** The Public Leaderboard F0.5 score (0.770) on our V2 pipeline was significantly lower than the local CV score (0.875). Additionally, our very first emergency submission (which mostly contained France data, leaving US/India empty) scored 0.153 simply by fluke, while a full unigram baseline scored 0.697. This initially suggested catastrophic overfitting to the 80/20 train/val split.
- **Resolution:** The public LB gap is primarily caused by a **zero-shot country (France)** in the test set that doesn't exist in the training set. The pure ML string-distance models memorized US/India specific patterns. To solve this, we moved to the **V3 Two-Stage Reranker**, leveraging a Transformer (Cross-Encoder) that inherently generalizes to zero-shot languages like French. We also implemented a LOCO (Leave-One-Country-Out) strategy to correctly mathematically bound our generalization error.

## 12. Pandas Int64Vector OOM during DataFrame Merges
- **Problem:** During the transition to the V3 pipeline, running `generate_candidates` and then calculating blocking recall on the full 100M+ pair dataset threw an `Int64Vector.resize MemoryError`. Pandas' `merge()` function attempts to allocate massive contiguous hash tables in memory, which immediately blew past 30GB of RAM even on standard string types.
- **Resolution:** Enforced **Strict Memory Chunking**. `compute_blocking_recall()` now slices candidates into explicitly bounded 5-million row batches before running `merge()`. `similarity.py`'s `loky` feature extractor was also rewritten to process strict batches and invoke `gc.collect()`, entirely resolving the MemoryError without needing to downscale the 1.0 training fraction.

## 13. Loky IPC Pickling OOM (54GB RAM Spike)
- **Problem:** Attempting to extract features across 14 cores using `loky` caused a massive OOM crash that hard-rebooted the server. We chunked the 205M pairs into 56 massive chunks. When the main thread mapped the IDs to 6 string arrays, it created an 18GB memory footprint. `loky` then attempted to serialize (pickle) this 18GB of strings to the IPC pipes, generating an additional 36GB memory spike, bringing total RAM to 54GB and crashing the OS.
- **Resolution:** Implemented **Micro-Chunking** in `similarity.py`. Sliced the 205M pairs into 500 tiny micro-chunks (~400,000 rows each). This keeps the active memory payload per batch strictly under 300MB, effortlessly sliding through the IPC pipes and locking the CPU at 100% (14 cores saturated) with near-zero memory footprint.

## 14. `np.vstack` Array Memory Error (13GB duplication)
- **Problem:** Feature extraction on the 205M rows ran flawlessly, but crashed at the very end when it tried to combine the 500 chunked arrays. Calling `np.vstack(results_list)` attempts to allocate a brand new 13.1 GB matrix and copy the 13.1 GB list into it, temporarily requiring 26.2 GB of contiguous memory, throwing an `_ArrayMemoryError`.
- **Resolution:** Eliminated `np.vstack`. Pre-allocated an empty matrix (`np.empty((len(pairs_df), 16), dtype=np.float32)`) *before* the multiprocessing loop, and dynamically wrote each micro-chunk directly into the matrix as it finished (`results_matrix[current_idx:current_idx+chunk_len] = chunk_arr`). This completely skips the 2x memory duplication.

## 15. LightGBM C++ `LGBM_BoosterCreate` Segfault
- **Problem:** Immediately after feature extraction, LightGBM threw an `OSError: exception: access violation writing 0x0000000000000000`. Python had successfully extracted the 17.5GB of training and validation arrays, but when `LGBMClassifier.fit()` was called, its C++ engine attempted to duplicate the arrays to build its internal histogram bins (`Dataset`). With Python already holding the 7GB Pandas dataframe cache, the OS ran out of pagefile space, causing `malloc()` to fail and crash C++.
- **Resolution:** Patched `main.py` with two memory optimizations:
  1. Explicitly deleted the 7GB Pandas cache (`del train_pairs`, `del data`) right before LightGBM training, and forced `gc.collect()`.
  2. Changed `joblib.load(..., mmap_mode='r')` for the cached features, mapping the 17.5GB matrices directly onto the SSD instead of RAM. Python memory dropped to 0GB, and LightGBM streamed the bins effortlessly.

## 16. O(N log N) Pandas Threshold Bottleneck & Caching Gap
- **Problem:** In Stage 6, the `sweep_threshold` loop repeatedly sorted and sliced the 20-million row Validation candidate dataset (primarily India) 66 times (once for each probability threshold). This O(66 * N log N) Pandas operation ran single-threaded and stalled the pipeline for >40 minutes per country, completely flatlining CPU utilization. Furthermore, because `main.py` only saved intermediate pipeline states at the very end of the file, killing the stalled threshold process resulted in the complete loss of 1.5 hours of Transformer Cross-Encoder GPU inferences (Stage 6.5).
- **Resolution:** 
  1. **Threshold Optimization:** Rewrote `threshold.py` to pre-sort the dataset by probability descending *once*. Replaced the Pandas sort and filter loop with an $O(\log N)$ binary search (`np.searchsorted`) to find the exact array slice index for each threshold, eliminating data copying. Handed the read-only slices to `joblib.Parallel(prefer="threads")` to evaluate all 66 thresholds concurrently across 14 cores (shared memory, zero serialization), reducing the bottleneck from 40 minutes down to <30 seconds.
  2. **Pipeline Caching:** Upgraded `main.py` to granularly cache `val_probs_reranked_{sample_key}.pkl` and `test_probs_reranked_{sample_key}.pkl` *immediately* after the Cross-Encoder finishes, ensuring expensive ML outputs are never lost due to downstream Pandas/I-O failures.

## Changelog
| Version | Date | By | Summary |
|---------|------|----|---------|
| v2.3 | 2026-09-26 | Antigravity | Added Issues 18 (Threshold-Dictionary TypeError) and 19 (V3 Cache Contamination). |
| v2.2 | 2026-09-26 | Antigravity | Added Issue 16 (O(N log N) Pandas Threshold Bottleneck & Caching Gap). |
| v2.1 | 2026-09-26 | Antigravity | Added Issues 13 (Loky IPC Pickling OOM), 14 (np.vstack Array Memory Error), and 15 (LightGBM Segfault / mmap). |
| v2.0 | 2026-09-26 | Antigravity | Added Issue 11 (Skewed Public LB due to Zero-shot France) and Issue 12 (Pandas Int64Vector OOM chunking fix). |
| v1.9 | 2026-09-26 | Antigravity | Added Issue 10 (RTX 5000 Series Blackwell PyTorch CUDA Incompatibility). |

## Issue 17: Polling / Sleeping Task Loops
- **Problem:** AI agents got stuck in task-management loops, constantly polling `status` and `cat`-ing logs of background tasks without yielding turns, which wasted >20 minutes of user time.
- **Solution:** Never use manual `sleep` timers or constant polling for background jobs. Launch the job, immediately yield control to the router, and let the system automatically wake the agent when the job finishes.

## Issue 18: Threshold-Dictionary Progress-Path TypeError
- **Problem:** In `main.py` line 273, the test inference progress file path was formatted as `f'test_progress_sample_{sample_key}_threshold_{best_threshold:.3f}.txt'`. When `sweep_threshold()` returns per-country thresholds as a `dict` (e.g. `{'US': 0.93, 'India': 0.92}`), Python's `:.3f` format spec raises `TypeError: unsupported format string passed to dict.__format__`. This prevented any V4 test inference from starting.
- **Resolution:** Replaced the float format with a conditional: if `best_threshold` is a dict, use the stable tag `"v4_percountry"`; otherwise use `f"{best_threshold:.3f}"`.

## Issue 19: V3 Cache Contamination in V4 Runs
- **Problem:** `model_cache_1.pkl` contained V3 artifacts: reranker-modified probabilities, V3 per-country thresholds (`US=0.93, India=0.95`), and V3 F0.5 (`0.8436`). Because `main.py` unconditionally loads this cache when it exists (`if os.path.exists(model_cache_path)`), any V4 run silently inherited V3 calibration. The V4 pure-ensemble thresholds are actually `US=0.93, India=0.92` — the India threshold differs by 0.03 due to the reranker's probability distortion.
- **Resolution:** Created an isolated `cache_v4/` directory with `v4_validate.py` that loads only the shared version-agnostic models (`lgb_model.pkl`, `xgb_model.pkl`) and computes fresh ensemble probabilities. The V4 manifest (`v4_manifest.json`) records exact provenance. A separate `v4_inference.py` reads exclusively from `cache_v4/`.
