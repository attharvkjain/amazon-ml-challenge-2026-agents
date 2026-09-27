> **Version:** v3.5 | **Last updated:** 2026-09-27 00:09 IST | **By:** Antigravity

# AI Agent Operating Manual

**If you are an AI coding agent working on this project, this is your operating manual.** Follow it exactly.

---

## Purpose

This file tells you how to orient yourself in this repo, what to read, what to do, and what to never do. It complements [`project.md`](project.md), which is the master index and single source of truth.

---

## Bootstrap Sequence

When you start working on any task for this project, follow these steps **in order**:

1. **Read [`project.md`](project.md)** — understand the project, check the Current State Snapshot, review the Master Index.
2. **Locate the relevant file(s)** from the Master Index table in `project.md` for your specific task.
3. **Read only the relevant file(s)** in `context/` for ordinary work. For reconciliation, read `project.md`, `agents.md`, and every file in `context/` in full before auditing.
4. **Check [`context/experiment-log.md`](context/experiment-log.md)** if your task involves modeling — someone may have already tried your approach.
5. **Check [`context/submission-log.md`](context/submission-log.md)** if your task involves a submission — never submit worse than the current best.
6. **Never assume.** If information you need isn't in the docs, say so explicitly. Ask the human — don't infer.

---

## Environment & Setup

> For full details, see [`context/environment-setup.md`](context/environment-setup.md).

### Quick Reference

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies (when requirements.txt exists)
pip install -r SUBMISSION/code/business_entity_resolution/requirements.txt
```

### Data Location

The dataset is stored locally in `data/6ab10eb3b23ba_student_resource/student_resource/dataset/` and is **not** committed to git. Each teammate has it at the same relative path. Key files:

- `dataset/train/train_source1.tsv` — Source 1 training records
- `dataset/train/train_source2.tsv` — Source 2 training records
- `dataset/train/train_source3.tsv` — Source 3 training records
- `dataset/train/train_ground_truth.tsv` — Ground truth labels
- `dataset/test/test_source1.tsv` — Source 1 test records
- `dataset/test/test_source2.tsv` — Source 2 test records
- `dataset/test/test_source3.tsv` — Source 3 test records

**All files are tab-separated.** Always use `sep='\t'`:

```python
import pandas as pd
df = pd.read_csv("data/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv", sep="\t")
```

---

## Validation Commands

Before considering any task done, confirm your changes didn't break the pipeline:

### 1. Validate Submission Format

```bash
python data/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching SUBMISSION/output/matching_results.tsv \
    --candidate SUBMISSION/output/candidate_pairs.tsv \
    --test-dir data/6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

Must print `PASS` (exit 0). If it fails, fix before proceeding.

### 2. Run Tests (when available)

```bash
# When unit tests exist in src/
python -m pytest SUBMISSION/code/business_entity_resolution/src/ -v
```

### 3. Check Output Shape

```python
import pandas as pd

# matching_results.tsv must have one row per S1 test entity.
# Check context/problem-and-data.md for canonical dataset statistics.
df = pd.read_csv("SUBMISSION/output/matching_results.tsv", sep="\t")
test_s1 = pd.read_csv("data/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv", sep="\t")
expected_rows = len(test_s1)
assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
```

---

## Agent Skills

The canonical, tracked Agent Skills live in `skills/`; provider directories mirror the canonical files for agent discovery; maintain skills in `skills/`. Use skills when requested or when appropriate:

- **`log-experiment`**: Logs a new experiment. Trigger when finishing a training run or explicitly asked.
- **`validate-submission`**: Validates a submission payload. Trigger before generating intermediate TSV files for the user, or before creating the final submission zip.
- **`eda-report`**: Logs an EDA report. Trigger when completing data exploration or asked to log findings.
- **`notebook-to-script`**: Extracts notebook logic to a script. Trigger when modularizing code or before committing a notebook.
- **`new-experiment`**: Scaffolds a new experiment. Trigger when starting a new approach.
- **`sync-writeup`**: Pulls the best score into the writeup draft. Trigger when asked to sync the methodology doc.
- **`reconcile-project`**: Audits documentation against the live repository, applies approved DOC-ONLY fixes, and gates CODE-ADJACENT changes on item-specific approval.

---

## Do's and Don'ts

### ✅ DO

- **Read `project.md` first** — every time, no exceptions
- **Append to `context/experiment-log.md`** when you run an experiment — add a new row at the bottom of the table
- **Append to `context/submission-log.md`** when a submission is made — add a new row at the bottom of the table
- **Update the version header** of any `.md` file you edit (bump minor version, add changelog entry)
- **Update the Current State Snapshot in `project.md`** if you achieve a new best score
- **Check the Current State Snapshot** before claiming a result is "better"
- **Use `sep='\t'`** for all data file I/O — files are tab-separated, not comma-separated
- **Leave a note** if you modify a teammate's work-in-progress file
- **Context Updation Rule:** Whenever the human explicitly states to "update the project context fully", you MUST systematically go through and update `agents.md`, `project.md`, and any relevant files in the `context/` directory to reflect the current state of the project.
- **Proactive Bottleneck Resolution:** If any pipeline stage takes an unreasonably long time, stop it immediately, identify the bottleneck (e.g., replace `iterrows` with `itertuples`, add `loky` multiprocessing), refactor the code, and restart. **CRITICAL:** Any performance rewrite must be verified to produce identical (or score-identical) output on a small sample before being trusted at full scale. Always proactively review code for performance bottlenecks before execution to ensure the fastest possible runtime given the goals.
- **Pipeline Caching:** Always save intermediate assets (models, extracted features, candidate pairs) to disk using `pickle` or `joblib`. If the pipeline crashes or is interrupted, reload from the latest checkpoint instead of recomputing from scratch.
- **Log Pitfalls:** Append any errors, crashes, bugs, performance bottlenecks, or tricky design issues you encounter and resolve to `context/challenges_faced.md`.
- **Optimize Candidate Sets (Final Evaluation Rule):** Always optimize blocking to produce the smallest possible candidate set while maintaining high recall. `candidate_pairs.tsv` is tracked and graded in the final ranking. The smaller the candidate set per S1 entity, the better.
- **Out-Of-Core Memory & Memory Limits (Critical Architecture Learnings):**
  - **Micro-Chunk Loky:** Never pass 10+ GB string arrays over `loky` IPC processes. Loky serializes `object` strings which triples the memory footprint (Main Thread + IPC Pipe + Worker Unpickle) and hard-crashes the OS. Pass small micro-chunks (max 300MB) when using `loky`.
  - **Memory-Mapped Caching:** When loading massive cached feature arrays (`.pkl`) for ML training, always use `joblib.load(..., mmap_mode='r')`. This maps the arrays directly to the SSD with zero Python RAM usage, leaving the entire system memory clear for C++ ML engines (LightGBM/XGBoost) to safely build their internal datasets.
  - **No `np.vstack` for massive matrices:** Never use `np.vstack()` on a massive list of Numpy arrays at the end of feature extraction (it duplicates the entire memory payload). Pre-allocate an `np.empty` matrix and copy chunks directly into it.
  - **Per-Model Caching:** Per-model caching is MANDATORY for ML Training stages to ensure a downstream model failure (e.g. XGBoost) doesn't wipe upstream progress (e.g. LightGBM).
  - **Single-Threaded Pandas Bottlenecks:** Never run Pandas `sort_values` or `drop_duplicates` inside a sequential loop on massive (20M+ row) dataframes (e.g., threshold sweeping). Pre-sort the dataset exactly once, and use $O(\log N)$ `np.searchsorted` parallelized via `joblib.Parallel(prefer="threads")` to evaluate slices instantly across all cores without memory duplication.
  - **Intermediate Pipeline Checkpointing:** Expensive ML operations (e.g., 1.5 hour Cross-Encoder inferences) MUST be explicitly dumped to `joblib` the exact moment they finish (`val_probs_reranked_{sample_key}.pkl`). Never wait until the end of the script to save state.

- **Apples-to-Apples Validation Standard:** All ad-hoc testing must use standardized data splits. For **Minor versions**, use a `7.5%` slice (`SAMPLE_FRAC = 0.075`). For **Major enhancements**, use an **`80%` slice (`SAMPLE_FRAC = 0.8`)** to measure candidate sets and recall.
- **Strict Submission Time Management:** To ensure the final run completes before the competition site crashes at 5:00 PM IST on Sept 27th:
  - **1) "Time to block for ultimate final submission run"**: To establish this baseline, run the full pipeline on a **Major (80%) slice**. Multiply that purely empirical time by `1.25` to estimate the 100% full run, then multiply by `1.75` for the final safety buffer. (e.g. `Time = Empirical_80_Time * 1.25 * 1.75`). **Current Limit**: `[PENDING V6.2 FINAL RUN]`
  - **2) "Development time"**: Calculated as (Time left until 5:00 PM Sept 27th) MINUS ("Time to block for ultimate final submission run"). **Current Value**: `[PENDING]`
  - *Agents must proactively update these variables upon completion of any milestone and explicitly warn the user if they grow stale.*

### ❌ DON'T

- **Never edit or delete existing rows** in `context/experiment-log.md` or `context/submission-log.md` — these are **append-only** logs
- **Never silently overwrite a better logged result** — only update the Current State Snapshot if the new score actually beats the existing best
- **Never touch raw data files** in `data/` — read-only access only
- **Never commit large files** (model weights, pickled objects, datasets) — they are in `.gitignore`
- **Never write one-time use things as files in the project root** (e.g. handoff prompts, quick calculations, ephemeral scripts) � provide them directly in chat or use a dedicated scratch/ folder that you clear periodically to prevent project bloat.
- **Never modify a teammate's in-progress notebook** without leaving a clearly visible note explaining what you changed and why
- **Never guess or infer** when information is missing — say "not in the docs" and ask
- **Never assume the data is comma-separated** — it's tab-separated
- **Never hardcode** the country list to `{US, India}` — the test set includes France (zero-shot)
- **Never skip the versioning rule** — see below
- Run reconciliation after every leaderboard submission, major architecture revision, and before final packaging or handoff; also run it when official rules or data paths change.

---

## Versioning Rule (Hard Requirement)

> **This is non-optional. Every `.md` file edit must follow this rule.**

Every markdown file starts with:

```
> **Version:** vX.Y | **Last updated:** YYYY-MM-DD HH:MM IST | **By:** <name>
```

Every markdown file ends with a `## Changelog` section (newest entry on top):

```markdown
| Version | Date | By | Summary |
|---------|------|----|---------|
| vX.Y | YYYY-MM-DD | Name | What changed |
```

For Agent Skill `SKILL.md` files, valid YAML frontmatter remains first. Put the standard version line immediately after the closing frontmatter fence and keep the changelog at the end.

**Any content edit bumps the version and adds a changelog line.** No exceptions.

---

## Escalation Rule

> **If the information you need isn't in the docs, ASK. Don't infer silently.**

Specifically:

- If a parameter, threshold, or design decision isn't documented → ask the human
- If you're unsure which approach is current best → check `project.md` Current State Snapshot, then ask if unclear
- If you find conflicting information between files → flag it, don't silently pick one
- If a file referenced in the Master Index doesn't exist or is empty → say so, don't fill it with guesses

---

## Changelog

| Version | Date | By | Summary |
|---------|------|----|---------|
| v3.5 | 2026-09-27 | Antigravity | Explicitly updated context after V4 LB result (0.788) per Context Updation Rule. Transitioned to V5 roadmap. |
| v3.4 | 2026-09-26 | Antigravity | Added Single-Threaded Pandas O(N log N) Bottlenecks and Intermediate Pipeline Checkpointing to Memory limits rule block. |
| v3.3 | 2026-09-26 | Antigravity | Added extensive Out-Of-Core Memory & Memory Limits rule block (loky chunks, mmap_mode, np.vstack, per-model caching). |
| v3.2 | 2026-09-26 | Antigravity | Added new DO rule to optimize Candidate Sets for final evaluation scoring. |
| v3.1 | 2026-09-26 | Codex | Clarified that provider skill folders are discovery mirrors of the canonical `skills/` directory. |
| v3.0 | 2026-09-26 | Codex | Added reconciliation bootstrap/trigger guidance and skill, corrected active lowercase data and canonical skills paths. |
| v2.0 | 2026-09-26 | Antigravity | Updated Agent Skills path, removed hardcoded test row count, and added verification requirement to Bottleneck rule. |
| v1.7 | 2026-09-25 | Antigravity | Added Log Pitfalls rule to record issues in challenges_faced.md |
| v1.6 | 2026-09-25 | Antigravity | Added Pipeline Caching rule to Do's to prevent lost work during crashes |
| v1.5 | 2026-09-25 | Antigravity | Added Proactive Bottleneck Resolution rule to ensure maximum performance and prompt interruption of slow runs |
| v1.4 | 2026-09-25 | Antigravity | Fixed test S1 row count assertion: 1,732,544 (verified) not 1,732,545 |
| v1.3 | 2026-09-25 | Antigravity | Added Agent Skills section and updated rules on not overwriting better scores. Fixed REPO path references to SUBMISSION. |
| v1.2 | 2026-09-25 | Antigravity | Added Context Updation Rule |
| v1.1 | 2026-09-25 | Member 1 | Updated paths to match submission package layout (SUBMISSION/code/business_entity_resolution/src/) |
| v1.0 | 2026-09-25 | Member 1 | Initial skeleton created |



