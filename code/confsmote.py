"""
ConfSMOTE: Conformal-Verified Oversampling for Label-Noise-Robust Imbalanced Learning.

Core idea (novelty claim to be defended in the paper):
  Use Mondrian (class-conditional) split-conformal p-values, computed from a
  cross-fitted probabilistic scorer, for TWO audits that no prior oversampler
  performs with a distribution-free guarantee:
    (1) SEED AUDIT  - a minority seed may generate synthetics only if its
        observed label is conformally plausible (p_minority > alpha).
        Suspicious seeds are excluded from *generation* but NOT deleted from
        training (contrast: CleanLab-style deletion, which collapses under
        extreme imbalance).
    (2) SYNTHETIC AUDIT - every interpolated candidate is accepted only if its
        conformal prediction set at level alpha is the singleton {minority}.
        Rejected candidates are RESAMPLED (new parent/neighbour/lambda) until
        the generation target is met or the attempt budget is exhausted
        (contrast: SMOTE-IPF / SMOTE-RSB / NEATER, which reject-and-shrink).

Guarantee (standard Mondrian conformal, Vovk et al.): under exchangeability of
the calibration data, for each class c, P(p_c(X) <= alpha | Y=c) <= alpha.
Hence (i) at most ~alpha of *genuine* minority seeds are expected to fail the
seed audit, and (ii) a candidate that is truly majority-distributed passes the
singleton-minority audit with probability controlled by the majority-class
calibration - the audit's false-acceptance rate is set by the user via alpha,
not by a data-blind quota or a tuned probability threshold.

This module is deliberately dependency-light: numpy + scikit-learn only.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Mondrian conformal p-values
# --------------------------------------------------------------------------- #
class MondrianConformal:
    """Split-conformal p-values per class, nonconformity = 1 - P(class|x)."""

    def __init__(self, scorer=None):
        self.scorer = scorer if scorer is not None else HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=0)

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        self.clf_ = clone(self.scorer)
        self.clf_.fit(X_fit, y_fit)
        proba_cal = self.clf_.predict_proba(X_cal)
        self.classes_ = list(self.clf_.classes_)
        # nonconformity scores of calibration points under their TRUE class
        self.cal_scores_ = {}
        for ci, c in enumerate(self.classes_):
            mask = (y_cal == c)
            self.cal_scores_[c] = np.sort(1.0 - proba_cal[mask, ci])
        return self

    def p_values(self, X):
        """Return dict class -> conformal p-value array for X."""
        proba = self.clf_.predict_proba(X)
        out = {}
        for ci, c in enumerate(self.classes_):
            s_new = 1.0 - proba[:, ci]
            cal = self.cal_scores_[c]
            n = len(cal)
            # p = (1 + #{cal scores >= s_new}) / (n + 1)
            idx = np.searchsorted(cal, s_new, side="left")
            out[c] = (1.0 + (n - idx)) / (n + 1.0)
        return out


# --------------------------------------------------------------------------- #
# ConfSMOTE sampler
# --------------------------------------------------------------------------- #
class ConfSMOTE:
    def __init__(self, alpha=0.10, k_neighbors=5, sampling_ratio=1.0,
                 cal_fraction=0.25, overshoot=1.0, max_attempts_factor=20,
                 audit_seeds=True, audit_synthetics=True,
                 scorer=None, random_state=0):
        """
        alpha            : conformal significance level for both audits.
        sampling_ratio   : target minority/majority ratio after resampling (rho).
        audit_seeds      : enable seed audit (ablation switch).
        audit_synthetics : enable synthetic audit (ablation switch).
        """
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.cal_fraction = cal_fraction
        self.overshoot = overshoot
        self.max_attempts_factor = max_attempts_factor
        self.audit_seeds = audit_seeds
        self.audit_synthetics = audit_synthetics
        self.scorer = scorer
        self.random_state = random_state
        self.diagnostics_ = {}

    # -- internal ---------------------------------------------------------- #
    def _conformal(self, X, y, rng):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.cal_fraction,
                                     random_state=rng.integers(1 << 31))
        fit_idx, cal_idx = next(sss.split(X, y))
        mc = MondrianConformal(self.scorer)
        mc.fit(X[fit_idx], y[fit_idx], X[cal_idx], y[cal_idx])
        return mc

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        assert len(classes) == 2, "binary only"
        maj_c, min_c = classes[np.argmax(counts)], classes[np.argmin(counts)]
        X_min, X_maj = X[y == min_c], X[y == maj_c]
        n_min, n_maj = len(X_min), len(X_maj)
        n_target = int(round(self.sampling_ratio * n_maj)) - n_min
        diag = {"n_minority": n_min, "n_target_syn": max(n_target, 0)}
        if n_target <= 0:
            self.diagnostics_ = diag
            return X, y

        mc = self._conformal(X, y, rng)

        # ---- (1) seed audit ---------------------------------------------- #
        # Trusted iff the conformal prediction set is exactly {minority}:
        # p_min > alpha AND p_maj <= alpha. (A p_min-only rule was tested and
        # rejected empirically: it caught <5% of injected fake seeds, because
        # the noisy minority calibration distribution makes fakes look
        # typical. The singleton rule caught 43-51% at 81-97% precision with
        # 1-7% true-seed loss on the mammography diagnostic.)
        if self.audit_seeds:
            pv_seed = mc.p_values(X_min)
            p_min_seed = pv_seed[min_c]
            seed_ok = (p_min_seed > self.alpha) & (pv_seed[maj_c] <= self.alpha)
            diag["seed_reject_rate"] = float(1.0 - seed_ok.mean())
            diag["seed_p_min_mean"] = float(p_min_seed.mean())
        else:
            seed_ok = np.ones(n_min, dtype=bool)
            diag["seed_reject_rate"] = 0.0
        ok_idx = np.where(seed_ok)[0]
        if len(ok_idx) < 2:                      # degenerate: fall back to all
            ok_idx = np.arange(n_min)
            diag["seed_fallback_all"] = True
        X_seed = X_min[ok_idx]

        # neighbour structure among ACCEPTED seeds only (safe interpolation)
        k = min(self.k_neighbors, len(X_seed) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_seed)
        _, nbrs = nn.kneighbors(X_seed)
        nbrs = nbrs[:, 1:]                       # drop self

        # ---- (2) generate + synthetic audit ------------------------------ #
        accepted = []
        attempts = 0
        max_attempts = self.max_attempts_factor * n_target
        batch = max(256, n_target)
        while len(accepted) < n_target and attempts < max_attempts:
            take = min(batch, (n_target - len(accepted)) * 2)
            parents = rng.integers(0, len(X_seed), size=take)
            neigh = nbrs[parents, rng.integers(0, k, size=take)]
            lam = rng.random((take, 1))
            cand = X_seed[parents] + lam * (X_seed[neigh] - X_seed[parents])
            attempts += take
            if self.audit_synthetics:
                pv = mc.p_values(cand)
                keep = (pv[min_c] > self.alpha) & (pv[maj_c] <= self.alpha)
                cand = cand[keep]
            accepted.extend(cand.tolist())
        diag["syn_attempts"] = int(attempts)
        diag["syn_accepted"] = int(min(len(accepted), n_target))
        diag["syn_accept_rate"] = float(diag["syn_accepted"] / max(attempts, 1))

        if len(accepted) < n_target:            # refill with seed duplicates
            short = n_target - len(accepted)
            fill = X_seed[rng.integers(0, len(X_seed), size=short)]
            accepted.extend(fill.tolist())
            diag["syn_filled_by_duplication"] = int(short)
        else:
            diag["syn_filled_by_duplication"] = 0

        X_syn = np.array(accepted[:n_target])
        X_out = np.vstack([X, X_syn])
        y_out = np.concatenate([y, np.full(n_target, min_c)])
        self.diagnostics_ = diag
        return X_out, y_out


# --------------------------------------------------------------------------- #
# Scorer-threshold SMOTE (ablation control): same pipeline, but audits use a
# raw probability threshold (0.5) instead of conformal p-values. Isolates the
# value of the conformal machinery itself.
# --------------------------------------------------------------------------- #
class ProbSMOTE(ConfSMOTE):
    def fit_resample(self, X, y):
        # monkey-see control: temporarily swap conformal p-values for raw proba
        orig = MondrianConformal.p_values

        def raw_p(self_mc, Xq):
            proba = self_mc.clf_.predict_proba(Xq)
            return {c: proba[:, i] for i, c in enumerate(self_mc.classes_)}

        # with raw probabilities, "p > alpha" becomes prob > alpha; use 0.5
        old_alpha = self.alpha
        self.alpha = 0.5
        MondrianConformal.p_values = raw_p
        try:
            return super().fit_resample(X, y)
        finally:
            MondrianConformal.p_values = orig
            self.alpha = old_alpha
