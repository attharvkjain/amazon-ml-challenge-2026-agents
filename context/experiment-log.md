> **Version:** v1.8 | **Last updated:** 2026-09-27 17:42 IST | **By:** Antigravity

# Experiment Log

**Standalone primer:** Amazon ML Challenge 2026 — Business Entity Resolution, evaluated on F₀.₅ (precision-heavy, macro-averaged). This file is the canonical record of every experiment/approach tried during the competition. It is used to track progress, prevent redundant work, and ensure we never lose track of what produced our best result.

---

> [!CAUTION]
> **APPEND-ONLY.** Never edit or delete existing rows. Only add new rows at the bottom. If a previous entry has an error, add a new row with a correction note — do not modify the original.

---

## How to Log an Experiment

Add a new row to the table below **before merging your branch to main**. Fill in every column.

---

## Experiment Log

| # | Date | Approach | Key Params | Local CV (F₀.₅) | Who | Commit | Notes |
|---|------|----------|------------|------------------|-----|--------|-------|
| 1 | 2026-09-25 | V1 TF-IDF Word-Unigram Baseline | max_df=0.01, K=20 | 0.9699 | Antigravity | `HEAD` | Scored 0.697 on Public LB. Too slow, sparse matrices too dense on char-ngrams. |
| 2 | 2026-09-26 | V2 GPU Semantic Blocking | K=10, MiniLM-L12 | pending | Antigravity | `HEAD` | Replacing TF-IDF to solve time limit and optimize for smaller candidate sets. |
| 3 | 2026-09-26 | V2 GPU Semantic Blocking (Completed) | K=10, MiniLM-L12, threshold=0.870 | 0.8755 | Antigravity | `HEAD` | Achieved blistering >97% precision, capped recall due to strict K=10 radius. |
| 4 | 2026-09-26 | V3 Two-Stage Pipeline (1.0 Split) | K=10, MiniLM-L12, LGBM, ms-marco-MiniLM-L-6-v2 (0.05-0.95), chunked-merges | pending | Antigravity | `HEAD` | Attempting to solve LB gap with a Transformer Reranker. Fixed Int64Vector OOM via 5M-row chunked batched merges. |
| 5 | 2026-09-26 | V3 Zero-Shot LOCO Simulation | K=10, LOCO=India, Forced Zero-Shot Mask=True | planned | Antigravity | `HEAD` | Will train strictly on US and evaluate strictly on India to mathematically simulate the France zero-shot gap, forcing 100% transformer reranking for zero-shot regions. |
| 6 | 2026-09-26 | V4 Pure Ensemble (Reproduced) | LGBM+XGB avg, K=25, word-unigram, US=0.930/India=0.920 | 0.8878 | Antigravity | `HEAD` | Clean V4 reproduction via v4_validate.py. No reranker, no penalty. Blocking recall=0.8108. 51.1M val pairs, 441K val S1. Manifest: cache_v4/v4_manifest.json. |
| 7 | 2026-09-27 | V5 Single-Pass Conditional Blocking | K=25 semantic, Orphan trigger < 0.75, Targeted 3-gram TF-IDF (top_k=5), Rare Hash (freq<=15) | pending CV | Antigravity | `HEAD` | Diagnostic proven: Isolated orphans (1.2% of dataset) and ran 90%-shrunken TF-IDF. Blocking recall jumped from 81.08% to 90.84% (+10% absolute). Time overhead: <4 mins CPU. Candidate volume: 63.4M (+12M). |
| 8 | 2026-09-27 | V6.1 Orphan-Targeted Bi-gram & Phonetic Blocking | S23 Orphan trigger < 0.85, Bi-gram Hash (freq<=100), Phonetic Metaphone Hash (freq<=50) | pending CV | Antigravity | `HEAD` | Solved Single-Threaded Pandas Bottleneck by replacing 1D token hashing with Bi-gram Hash Blocking. Ran cleanly on CPU in <5 mins without OOM. Candidate volume: 119M (+67.9M). Blocking recall jumped from 81.08% to 96.75%. |
| 9 | 2026-09-27 | V6.2 Global Numeric Hash | Added targeted numeric extraction (digits stripped of leading zeros, `freq <= 500`) globally across S23. Reverted K=25. | pending CV | Antigravity | `HEAD` | Caught heavily transliterated S2 strings where numbers survived exactly. Prevented `[1, 0, 01]` common-number memory explosion. Recall jumped from 96.75% to 99.43%. Extrapolated candidates: 135M. |

| 10 | 2026-09-27 | V6.2 Full 80/20 Evaluation Run | LGBM+XGB avg, K=25 semantic, 5 CPU heuristics (TF-IDF/Rare/Bigram/Phonetic/Numeric), per-country thresholds India=0.930/US=0.950 | 0.9478 | Antigravity | `HEAD` | First complete end-to-end run of V6.2. Blocking recall=95.50%. 145M India pairs, ~90M US pairs (est). Per-country F0.5: India=0.9288, US=0.9605. Singleton accuracy=90.42%. Test inference in progress for Public LB submission. |

*(Add new experiments above this line.)*

---

## Quick Reference

- **Current best local CV**: See `project.md` → Current State Snapshot
- **Current best leaderboard**: See [`context/submission-log.md`](submission-log.md)
- **Problem details**: See [`context/problem-and-data.md`](problem-and-data.md)

---

## Changelog
| Version | Date | By | Summary |
|---------|------|----|---------|
| v1.8 | 2026-09-27 | Antigravity | Logged experiment #10: V6.2 Full 80/20 Evaluation Run with 0.9478 overall F0.5. |
| v1.7 | 2026-09-27 | Antigravity | Logged experiment #9: V6.2 Global Numeric Hash Blocking yielding 99.43% candidate recall. |
| v1.6 | 2026-09-27 | Antigravity | Logged experiment #8: V6.1 Orphan-Targeted Bi-gram & Phonetic Blocking. |
| v1.5 | 2026-09-27 | Antigravity | Logged experiment #7: V5 Single-Pass Conditional Blocking. |
| v1.4 | 2026-09-26 | Antigravity | Logged experiment #6: V4 Pure Ensemble reproduced at 0.8878 with clean cache provenance. |
| v1.3 | 2026-09-26 | Antigravity | Logged pending V3 Two-Stage and planned V3 LOCO experiments. |
| v1.2 | 2026-09-26 | Antigravity | Logged completed results for V2 GPU Semantic Blocking. |
| v1.1 | 2026-09-26 | Antigravity | Logged V1 Baseline and pending V2 GPU Semantic Blocking experiments. |
| v1.0 | 2026-09-25 | Member 1 | Initial skeleton created |
| 15 | LGBM + XGB Ensemble | None | LightGBM/XGBoost ensemble evaluated directly (Stage 1 only). Discovered Transformer Reranker was degrading score by 0.04. | 0.8878 | Drop Transformer Reranker entirely, return to V2 Ensemble Architecture. |
