"""Shared utilities: datasets, label-noise injection with KNOWN ground-truth
masks (so gate quality can be measured, not assumed), and metrics."""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Noise injection. All rates are PER-CLASS and printed in the paper.
# Returns (y_noisy, fake_seed_mask) where fake_seed_mask marks training points
# whose observed minority label is FALSE (only meaningful for contamination).
# --------------------------------------------------------------------------- #

def inject_contamination(X, y, n_fake, rng):
    """Flip `n_fake` majority points to minority (false-fraud seeds).
    Simulates false alarms / dispute-tagging errors (0 -> 1)."""
    y2 = y.copy()
    maj_idx = np.where(y == 0)[0]
    chosen = rng.choice(maj_idx, size=min(n_fake, len(maj_idx)), replace=False)
    y2[chosen] = 1
    fake_mask = np.zeros(len(y), dtype=bool)
    fake_mask[chosen] = True
    return y2, fake_mask


def inject_minority_loss(y, eta, rng):
    """Flip a fraction eta of true minority points to majority (1 -> 0).
    Simulates undetected / late-reported fraud."""
    y2 = y.copy()
    min_idx = np.where(y == 1)[0]
    n_flip = int(round(eta * len(min_idx)))
    chosen = rng.choice(min_idx, size=n_flip, replace=False)
    y2[chosen] = 0
    return y2


# --------------------------------------------------------------------------- #
# Metrics (threshold-dependent ones at FIXED 0.5 for every method)
# --------------------------------------------------------------------------- #
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             f1_score, recall_score, balanced_accuracy_score,
                             roc_auc_score)


def score_all(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    out = {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
    }
    return out
