"""
Evaluation — F0.5 scorer, CV harness, and diagnostic plots.

Per architecture.md §6, mandatory diagnostics:
  1. Blocking recall
  2. Feature importance bar chart
  3. Score vs threshold curve (F0.5, precision, recall)
  4. Confusion matrix heatmap
  5. Per-country F0.5 table + bar chart
  6. Score distribution histogram (true matches vs non-matches)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DIAGNOSTICS_DIR, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP


def f05_per_entity(predicted: set, true: set) -> float:
    """F0.5 for a single S1 entity."""
    if not true and not predicted:
        return 1.0  # Singleton with empty prediction
    if not true and predicted:
        return 0.0  # Singleton with false prediction
    if true and not predicted:
        return 0.0  # Missed all matches

    tp = len(predicted & true)
    fp = len(predicted - true)
    fn = len(true - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta = 0.5
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def f05_macro(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str] | None = None,
) -> float:
    """Macro-averaged F0.5 across all S1 entities."""
    if all_s1_ids is None:
        all_s1_ids = set(ground_truth.keys())

    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(f05_per_entity(pred, true))

    return float(np.mean(scores)) if scores else 0.0


def compute_diagnostics(
    pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str],
    feature_names: list[str],
    feature_importances: dict[str, float],
    threshold: float,
    blocking_recall: float,
    save_dir: str | os.PathLike | None = None,
) -> dict:
    """
    Generate all mandatory diagnostic outputs per architecture.md §6.

    Returns dict of diagnostic values for logging.
    """
    if save_dir is None:
        save_dir = DIAGNOSTICS_DIR
    os.makedirs(save_dir, exist_ok=True)

    diagnostics = {}

    # 1. Blocking recall (already computed, just log it)
    diagnostics['blocking_recall'] = blocking_recall
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Blocking recall: {blocking_recall:.4f}")

    # 2. Feature importance bar chart
    _plot_feature_importance(feature_importances, save_dir)

    # 3. Score vs threshold curve
    _plot_threshold_curve(pairs_df, probabilities, ground_truth, all_s1_ids, threshold, save_dir)

    # 4. Confusion matrix at chosen threshold
    _plot_confusion_matrix(probabilities, labels, threshold, save_dir)

    # 5. Per-country F0.5
    country_scores = _per_country_scores(pairs_df, probabilities, ground_truth, all_s1_ids, threshold)
    diagnostics['per_country_f05'] = country_scores
    print(f"\n  Per-country F0.5:")
    for country, score in country_scores.items():
        print(f"    {country}: {score:.4f}")

    # 6. Score distribution histogram
    _plot_score_distribution(probabilities, labels, save_dir)

    # Overall F0.5 at threshold
    from postprocessing.threshold import _apply_threshold_and_constraint
    predictions = _apply_threshold_and_constraint(pairs_df, probabilities, threshold)
    overall_f05 = f05_macro(predictions, ground_truth, all_s1_ids)
    diagnostics['overall_f05'] = overall_f05
    t_str = str({k: f"{v:.3f}" for k, v in threshold.items()}) if isinstance(threshold, dict) else f"{threshold:.3f}"
    print(f"\n  Overall F0.5 at threshold {t_str}: {overall_f05:.4f}")

    # Singleton accuracy
    n_singletons = sum(1 for s1 in all_s1_ids if not ground_truth.get(s1, set()))
    n_singleton_correct = sum(
        1 for s1 in all_s1_ids
        if not ground_truth.get(s1, set()) and s1 not in predictions
    )
    if n_singletons > 0:
        singleton_acc = n_singleton_correct / n_singletons
        diagnostics['singleton_accuracy'] = singleton_acc
        print(f"  Singleton accuracy: {n_singleton_correct}/{n_singletons} = {singleton_acc:.4f}")

    print(f"{'='*60}\n")

    return diagnostics


def _plot_feature_importance(
    importances: dict[str, float],
    save_dir: str | os.PathLike,
) -> None:
    """Save feature importance bar chart."""
    names = list(importances.keys())
    values = list(importances.values())

    # Sort by importance
    sorted_indices = np.argsort(values)
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (LightGBM)')
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_importance.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_threshold_curve(
    pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str],
    best_threshold: float,
    save_dir: str | os.PathLike,
) -> None:
    """Save F0.5 / precision / recall vs threshold curve."""
    from postprocessing.threshold import _apply_threshold_and_constraint

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP * 5)
    f05s, precisions, recalls = [], [], []

    for t in thresholds:
        predictions = _apply_threshold_and_constraint(pairs_df, probabilities, t)

        # Compute precision and recall
        tp_total, fp_total, fn_total = 0, 0, 0
        f05_scores = []
        for s1_id in all_s1_ids:
            pred = predictions.get(s1_id, set())
            true = ground_truth.get(s1_id, set())
            tp = len(pred & true)
            fp = len(pred - true)
            fn = len(true - pred)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            f05_scores.append(f05_per_entity(pred, true))

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

        f05s.append(np.mean(f05_scores))
        precisions.append(prec)
        recalls.append(rec)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f05s, label='F0.5 (macro)', linewidth=2)
    ax.plot(thresholds, precisions, label='Precision (micro)', linestyle='--')
    ax.plot(thresholds, recalls, label='Recall (micro)', linestyle='--')
    
    t_str = str({k: f"{v:.3f}" for k, v in best_threshold.items()}) if isinstance(best_threshold, dict) else f"{best_threshold:.3f}"
    plot_t = list(best_threshold.values())[0] if isinstance(best_threshold, dict) else best_threshold
    ax.axvline(x=plot_t, color='red', linestyle=':', label=f'Best threshold = {t_str}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Score vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'threshold_curve.png')
    for _ in range(3):
        try:
            fig.savefig(path, dpi=100)
            break
        except OSError:
            import time
            time.sleep(1)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_confusion_matrix(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float | dict,
    save_dir: str | os.PathLike,
) -> None:
    """Save confusion matrix heatmap."""
    plot_t = list(threshold.values())[0] if isinstance(threshold, dict) else threshold
    preds = (probabilities >= plot_t).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    print(f"  Confusion matrix: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=14)
    t_str = str({k: f"{v:.3f}" for k, v in threshold.items()}) if isinstance(threshold, dict) else f"{threshold:.3f}"
    ax.set_title(f'Confusion Matrix (threshold={t_str})')
    fig.colorbar(im)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    for _ in range(3):
        try:
            fig.savefig(path, dpi=100)
            break
        except OSError:
            time.sleep(1)
    plt.close(fig)
    print(f"  Saved: {path}")


def _per_country_scores(
    pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str],
    threshold: float,
) -> dict[str, float]:
    """Compute per-country F0.5 scores."""
    from postprocessing.threshold import _apply_threshold_and_constraint

    # Build s1_id -> country mapping from pairs (vectorized)
    s1_country = pairs_df[['s1_id', 'country']].drop_duplicates().set_index('s1_id')['country'].to_dict()

    predictions = _apply_threshold_and_constraint(pairs_df, probabilities, threshold)

    countries = sorted(set(s1_country.values()))
    country_scores = {}
    for country in countries:
        country_s1_ids = {s1 for s1, c in s1_country.items() if c == country}
        # Also include S1 IDs from all_s1_ids that aren't in pairs but are from this country
        scores = []
        for s1_id in country_s1_ids & all_s1_ids:
            pred = predictions.get(s1_id, set())
            true = ground_truth.get(s1_id, set())
            scores.append(f05_per_entity(pred, true))
        if scores:
            country_scores[country] = float(np.mean(scores))

    # Save per-country bar chart
    if country_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(country_scores.keys(), country_scores.values())
        ax.set_ylabel('F0.5')
        ax.set_title('F0.5 by Country')
        ax.set_ylim(0, 1)
        for i, (c, v) in enumerate(country_scores.items()):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.tight_layout()
        path = os.path.join(DIAGNOSTICS_DIR, 'per_country_f05.png')
        for _ in range(3):
            try:
                fig.savefig(path, dpi=100)
                break
            except OSError:
                import time
                time.sleep(1)
        plt.close(fig)
        print(f"  Saved: {path}")

    return country_scores


def _plot_score_distribution(
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_dir: str | os.PathLike,
) -> None:
    """Save overlapping histogram of match probabilities for pos/neg."""
    pos_probs = probabilities[labels == 1]
    neg_probs = probabilities[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(neg_probs, bins=50, alpha=0.5, label=f'Non-matches ({len(neg_probs):,})', density=True)
    ax.hist(pos_probs, bins=50, alpha=0.5, label=f'True matches ({len(pos_probs):,})', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution: True Matches vs Non-Matches')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'score_distribution.png')
    import time
    for _ in range(3):
        try:
            fig.savefig(path, dpi=100)
            break
        except OSError:
            time.sleep(1)
    plt.close(fig)
    print(f"  Saved: {path}")
