"""
CrossConfSMOTE (v2): cross-conformal audited oversampling.

Why v2: split-conformal calibration for the minority class is tiny under
extreme imbalance (e.g. ~50 of 11k points on mammography, ~7 on yeast_ml8),
so split-conformal p-values are coarse and the audit is weak. Cross-conformal
auditing fixes the calibration size:
  * Train K scorers, each on (K-1)/K of the training data.
  * Every training point receives an out-of-fold (OOF) nonconformity score
    from the scorer that never saw it. Pooling OOF scores per class gives a
    calibration distribution built from ALL points of that class.
  * SEED AUDIT: a minority seed is trusted iff its OOF conformal prediction
    set is exactly {minority}.
  * SYNTHETIC AUDIT: a candidate's p-values are computed under every fold
    scorer (against the pooled OOF calibration) and combined by the median;
    accepted iff the combined prediction set is exactly {minority};
    rejected candidates are resampled until the target is met.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


def _pvals_from_cal(cal_scores: np.ndarray, s_new: np.ndarray) -> np.ndarray:
    cal = np.sort(cal_scores)
    n = len(cal)
    idx = np.searchsorted(cal, s_new, side="left")
    return (1.0 + (n - idx)) / (n + 1.0)


class CrossConfSMOTE:
    def __init__(self, alpha=0.10, k_neighbors=5, sampling_ratio=1.0,
                 n_folds=5, audit_seeds=True, audit_synthetics=True,
                 scorer=None, random_state=0):
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.n_folds = n_folds
        self.audit_seeds = audit_seeds
        self.audit_synthetics = audit_synthetics
        self.scorer = scorer if scorer is not None else HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=0)
        self.random_state = random_state
        self.diagnostics_ = {}

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        maj_c, min_c = classes[np.argmax(counts)], classes[np.argmin(counts)]
        X_min = X[y == min_c]; n_min, n_maj = int(counts.min()), int(counts.max())
        n_target = int(round(self.sampling_ratio * n_maj)) - n_min
        diag = {"n_minority": n_min, "n_target_syn": max(n_target, 0)}
        if n_target <= 0:
            self.diagnostics_ = diag; return X, y

        # ---- cross-conformal OOF scores ---------------------------------- #
        skf = StratifiedKFold(self.n_folds, shuffle=True,
                              random_state=self.random_state)
        oof_proba = np.zeros((len(y), len(classes)))
        fold_models = []
        for tr_idx, ho_idx in skf.split(X, y):
            m = clone(self.scorer); m.fit(X[tr_idx], y[tr_idx])
            fold_models.append(m)
            oof_proba[ho_idx] = m.predict_proba(X[ho_idx])
        ci = {c: i for i, c in enumerate(classes)}
        # calibration: OOF nonconformity under TRUE class, pooled per class
        cal = {c: 1.0 - oof_proba[y == c, ci[c]] for c in classes}
        # each point's nonconformity under EACH class (from its OOF model)
        nc = {c: 1.0 - oof_proba[:, ci[c]] for c in classes}

        # ---- (1) seed audit ---------------------------------------------- #
        min_mask = (y == min_c)
        if self.audit_seeds:
            p_min = _pvals_from_cal(cal[min_c], nc[min_c][min_mask])
            p_maj = _pvals_from_cal(cal[maj_c], nc[maj_c][min_mask])
            seed_ok_all = (p_min > self.alpha) & (p_maj <= self.alpha)
            diag["seed_reject_rate"] = float(1.0 - seed_ok_all.mean())
        else:
            seed_ok_all = np.ones(n_min, dtype=bool); diag["seed_reject_rate"] = 0.0
        ok_idx = np.where(seed_ok_all)[0]
        if len(ok_idx) < 2:
            ok_idx = np.arange(n_min); diag["seed_fallback_all"] = True
        X_seed = X_min[ok_idx]

        k = min(self.k_neighbors, len(X_seed) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_seed)
        _, nbrs = nn.kneighbors(X_seed); nbrs = nbrs[:, 1:]

        # ---- (2) generate + synthetic audit ------------------------------ #
        accepted, attempts = [], 0
        max_attempts = 20 * n_target
        batch = max(256, n_target)
        while len(accepted) < n_target and attempts < max_attempts:
            take = min(batch, (n_target - len(accepted)) * 2)
            parents = rng.integers(0, len(X_seed), size=take)
            neigh = nbrs[parents, rng.integers(0, k, size=take)]
            lam = rng.random((take, 1))
            cand = X_seed[parents] + lam * (X_seed[neigh] - X_seed[parents])
            attempts += take
            if self.audit_synthetics:
                p_min_folds, p_maj_folds = [], []
                for m in fold_models:
                    pr = m.predict_proba(cand)
                    p_min_folds.append(_pvals_from_cal(cal[min_c], 1.0 - pr[:, ci[min_c]]))
                    p_maj_folds.append(_pvals_from_cal(cal[maj_c], 1.0 - pr[:, ci[maj_c]]))
                p_min = np.median(np.array(p_min_folds), axis=0)
                p_maj = np.median(np.array(p_maj_folds), axis=0)
                cand = cand[(p_min > self.alpha) & (p_maj <= self.alpha)]
            accepted.extend(cand.tolist())
        diag["syn_attempts"] = int(attempts)
        diag["syn_accepted"] = int(min(len(accepted), n_target))
        diag["syn_accept_rate"] = float(diag["syn_accepted"] / max(attempts, 1))
        if len(accepted) < n_target:
            short = n_target - len(accepted)
            accepted.extend(X_seed[rng.integers(0, len(X_seed), size=short)].tolist())
            diag["syn_filled_by_duplication"] = int(short)
        else:
            diag["syn_filled_by_duplication"] = 0

        X_syn = np.array(accepted[:n_target])
        self.diagnostics_ = diag
        return (np.vstack([X, X_syn]),
                np.concatenate([y, np.full(n_target, min_c)]))
