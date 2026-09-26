> **Version:** v1.4 | **Last updated:** 2026-09-26 22:07 IST | **By:** Antigravity

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
| v1.4 | 2026-09-26 | Antigravity | Logged experiment #6: V4 Pure Ensemble reproduced at 0.8878 with clean cache provenance. |
| v1.3 | 2026-09-26 | Antigravity | Logged pending V3 Two-Stage and planned V3 LOCO experiments. |
| v1.2 | 2026-09-26 | Antigravity | Logged completed results for V2 GPU Semantic Blocking. |
| v1.1 | 2026-09-26 | Antigravity | Logged V1 Baseline and pending V2 GPU Semantic Blocking experiments. |
| v1.0 | 2026-09-25 | Member 1 | Initial skeleton created |
| 15 | LGBM + XGB Ensemble | None | LightGBM/XGBoost ensemble evaluated directly (Stage 1 only). Discovered Transformer Reranker was degrading score by 0.04. | 0.8878 | Drop Transformer Reranker entirely, return to V2 Ensemble Architecture. |
