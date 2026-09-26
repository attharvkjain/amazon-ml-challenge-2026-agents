> **Version:** v1.1 | **Last updated:** 2026-09-26 20:38 IST | **By:** Codex

# V4 Handoff


## Verified State

- Public baseline: Submission 3, F0.5 = 0.770.
- Documented best local score: pure LightGBM + XGBoost ensemble, F0.5 = 0.8878, with the transformer bypassed.
- The previously cached V3 result is F0.5 = 0.8436 with thresholds `US=0.93` and `India=0.95`; those artifacts must not be treated as V4 calibration.
- `main.py` still reuses `model_cache_1.pkl` when it exists, so a V4 run may silently inherit V3 validation probabilities and thresholds.
- `main.py` also formats a per-country threshold dictionary as a float in the progress-file name. This must be fixed before a clean V4 inference run.
- No reproducible artifact currently ties the 0.8878 score to its exact candidates, thresholds, post-processing, and cached model state.

## Required Work, In Order

1. Preserve the finished inference output under a versioned V4 filename. Run the official validator against that copy. Do not overwrite Submission 3.
2. Record whether the run completed, failed, or used stale V3 cache state. If it failed at the dictionary-format error, preserve the traceback as a diagnostic.
3. Make V4 cache-safe:
   - use a V4-specific experiment ID and cache directory;
   - do not load V3 `model_cache_1.pkl`, reranked probabilities, or V3 thresholds;
   - persist pure-ensemble probabilities, per-country thresholds, F0.5, candidate recall, candidate volume, and exact configuration in a V4 manifest;
   - replace the float threshold formatting in progress paths with the V4 experiment ID or a stable hash.
4. Reproduce the pure-ensemble validation score from clean V4 artifacts. Log the exact result in the existing experiment table with date, configuration, score, and notes. The log is append-only.
5. Run pure V4 inference into versioned outputs and validate the payload. Treat it as the next submission candidate only if the run is reproducible and valid.
6. After V4 is stable, measure candidate recall by country and source, then classify false negatives into: absent candidate, threshold rejection, one-to-one conflict, or post-processing rejection.
7. Only if candidate recall is below the score target, add compact candidate paths: lexical typo retrieval, address-led retrieval, name-led retrieval, and S2-to-S3 corroboration. Keep each path only when it recovers labeled matches.

## Non-Negotiable Constraints

- Do not interrupt any running process or agent task.
- Do not delete, reset, checkout, or overwrite existing caches, outputs, or in-progress edits.
- Do not use external identity data or submit to the leaderboard.
- Keep work isolated from the active pipeline until the existing run is complete.
- Follow `agents.md`: cache expensive work, use tab-separated data I/O, preserve raw data, validate generated TSVs, and version every Markdown edit.

## Success Criteria

- A clean, repeatable V4 validation result with complete provenance.
- A valid versioned V4 test output that did not reuse V3 calibration.
- Candidate recall and error-cause metrics available before further architecture changes.
- The 0.770 public submission remains recoverable.

## How to Pursue 0.99

### Reality Check

The documented pure-ensemble local score of 0.8878 is a useful recovery from the transformer regression, but it is not evidence that 0.99 is close. It is still 0.0922 below 0.98 and the prior V2 system lost about 0.1055 between local validation (0.8755) and public leaderboard (0.770). No threshold adjustment or LightGBM hyperparameter sweep can credibly close that gap alone.

A score near 0.99 is possible only if diagnostics uncover a large missing signal: true matches that are absent from the current candidate list, deterministic name/address transformations that can be retrieved more reliably, or evidence shared across multiple S2/S3 variants of the same S1 entity. The leading teams may have such a signal, but their methods are not known from the leaderboard.

### The Score Ceiling Comes First

The candidate generator sets the maximum attainable recall. If a true S2/S3-to-S1 pair is absent from the candidate set, no classifier, reranker, or rule can recover it. Before adding another model, measure candidate recall by country and source:

| Metric | Why it matters | Gate for a 0.99 attempt |
|--------|----------------|--------------------------|
| Candidate recall | Hard upper bound on matching recall | Approximately 99.5% or higher |
| Candidate volume | Affects final candidate-set ranking and runtime | Keep the smallest set that preserves recall |
| False-positive rate | F0.5 punishes wrong merges heavily | Must be extremely low on hard negatives |
| Transfer score | Proxy for France zero-shot behavior | Must not collapse on US-to-India or India-to-US |

If candidate recall is materially below the target, stop tuning the classifier and improve retrieval.

### Phase A: Diagnose the Existing System

For each validation false negative, assign exactly one cause:

1. The true pair was not retrieved.
2. It was retrieved but fell below the acceptance threshold.
3. It was replaced by another S1 due to the one-to-one S2/S3 constraint.
4. It was removed by a post-processing rule.

For each false positive, categorize the collision: same or nearby address with different businesses, related legal/company names, generic-name collision, number conflict, or another repeatable pattern. Do not add a rule until a category is measured and recurring.

### Phase B: Recover Missing Candidates

Keep country as a hard partition. Build a deduplicated union of compact retrieval paths and report the marginal labeled-match recovery and added candidate volume for each path:

1. Current multilingual semantic retrieval.
2. Normalized exact anchors for high-confidence name/address variants.
3. Name-led retrieval for records with useful names and weak/missing addresses.
4. Address-led retrieval for aliases, domains, and severe name changes with stable locations.
5. Lexical typo/abbreviation retrieval that is bounded by rare tokens or blocking keys, avoiding the previous exhaustive character-TFIDF runtime failure.
6. High-confidence S2-to-S3 links, used as an extra route to the same S1 candidate.

Remove any path that increases volume without recovering labeled true pairs. The aim is not more candidates; it is more true pairs per candidate.

### Phase C: Use Multi-Record Evidence

The current matcher judges one S2/S3 record against one S1 record in isolation. It discards a useful structural fact: several S2/S3 records can describe the same S1 business.

1. Seed only very high-confidence S2/S3-to-S1 assignments.
2. Link other S2/S3 variants to those seeds when they share strong address, rare-token, or semantic evidence.
3. Use this corroboration as a feature or conservative second-pass decision for difficult aliases.
4. Preserve the rule that every S2/S3 ID maps to at most one S1 ID.
5. Measure the gain and false-positive cost separately; do not propagate weak or circular matches.

### Phase D: Make the Pair Judge Harder to Fool

Use the existing LightGBM/XGBoost ensemble as the benchmark. Train or tune only on top of the improved candidate set.

- Mine hard negatives from records with the same address, similar names, or close semantic neighbors that are known not to match.
- Add features for rare-token agreement, full number-set agreement, address/name agreement separately, source, S2/S3 corroboration, and the score margin to the next-best S1 candidate.
- Apply a number-conflict veto only after measuring how many true matches it removes.
- Do not reintroduce the generic MS MARCO reranker. If a cross-encoder is reconsidered, it must be trained or calibrated on labeled business pairs and demonstrate an incremental gain on held-out and transfer validation.

### Phase E: France and Final Decisions

France has no labels. Do not claim it is validated. Use the weaker of US-to-India and India-to-US transfer experiments to choose a conservative France threshold and to test whether retrieval, not model memorization, is carrying the system.

Before replacing the baseline output or considering a submission, require:

1. A reproducible clean V4 score with saved configuration and artifacts.
2. Candidate recall and volume by country/source.
3. A score improvement on the same validation split as V2.
4. No material collapse on both transfer experiments.
5. Official payload validation passing on versioned outputs.

### Time Allocation

- First: make pure V4 reproducible and produce one valid, versioned output.
- Next: spend the shortest possible time measuring candidate recall and error causes.
- Then: invest in retrieval or S2/S3 corroboration only when diagnostics show it can recover a large class of misses.
- Do not spend remaining time on generic transformer experiments, broad hyperparameter sweeps, or undocumented rules.

## Changelog

| Version | Date | By | Summary |
|---------|------|----|---------|
| v1.1 | 2026-09-26 | Codex | Added the full recovery roadmap toward a 0.99 leaderboard score. |
| v1.0 | 2026-09-26 | Codex | Created the post-inference V4 handoff artifact. |
