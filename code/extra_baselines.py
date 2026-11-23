"""
Labelled reimplementations of the three must-beat prior methods from the
novelty audit. Original code is not publicly available for all three, so
these implement the published algorithms' core mechanisms as faithfully as
practical; the manuscript must describe them as reimplementations.

  * SMOTE-IPF (Saez et al. 2015): SMOTE, then iterative partition filtering
    with a bagged-tree ensemble; points the ensemble majority disagrees with
    are removed; iterate until the removed fraction stabilises.
  * RSMOTE (2021): minority points classified SAFE / BORDERLINE / NOISE by
    relative density (mean distance to minority neighbours over mean
    distance to majority neighbours); noisy seeds excluded; borderline
    seeds use a compressed interpolation range.
  * GK-SMOTE (2025): Gaussian-kernel density estimation over the minority
    class; seeds sampled with probability proportional to local density;
    synthetic points drawn by Gaussian perturbation around seeds and kept
    only where minority density dominates majority density.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def _split_classes(X, y):
    classes, counts = np.unique(y, return_counts=True)
    maj = classes[np.argmax(counts)]; minc = classes[np.argmin(counts)]
    return maj, minc, int(counts.max()), int(counts.min())


class SMOTEIPF:
    def __init__(self, random_state=0, n_iterations=3, n_partitions=9):
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.n_partitions = n_partitions
        self.diagnostics_ = {}

    def fit_resample(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        Xr, yr = SMOTE(random_state=self.random_state,
                       k_neighbors=5).fit_resample(X, y)
        Xr, yr = np.asarray(Xr), np.asarray(yr)
        for _ in range(self.n_iterations):
            n = len(yr)
            idx = rng.permutation(n)
            parts = np.array_split(idx, self.n_partitions)
            votes = np.zeros(n); seen = np.zeros(n)
            for p in parts:
                rest = np.setdiff1d(idx, p)
                tree = DecisionTreeClassifier(random_state=self.random_state)
                tree.fit(Xr[rest], yr[rest])
                votes[p] += (tree.predict(Xr[p]) != yr[p])
                seen[p] += 1
            bad = (votes > 0.5 * np.maximum(seen, 1))
            if bad.mean() < 0.01 or bad.all():
                break
            Xr, yr = Xr[~bad], yr[~bad]
        self.diagnostics_ = {"ipf_remaining": int(len(yr))}
        return Xr, yr


class RSMOTEApprox:
    def __init__(self, random_state=0, k_neighbors=5, sampling_ratio=1.0):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.diagnostics_ = {}

    def fit_resample(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        maj, minc, n_maj, n_min = _split_classes(X, y)
        n_target = int(round(self.sampling_ratio * n_maj)) - n_min
        if n_target <= 0:
            return X, y
        X_min, X_maj = X[y == minc], X[y == maj]
        k = min(self.k_neighbors, n_min - 1)
        d_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min) \
            .kneighbors(X_min)[0][:, 1:].mean(axis=1)
        d_maj = NearestNeighbors(n_neighbors=min(k, n_maj)).fit(X_maj) \
            .kneighbors(X_min)[0].mean(axis=1)
        rd = d_min / np.maximum(d_maj, 1e-12)     # relative density proxy
        safe = rd <= np.median(rd)                  # denser-in-minority half
        noise = rd > np.quantile(rd, 0.9)           # sparsest decile = noise
        seeds = X_min[~noise]; seed_safe = safe[~noise]
        if len(seeds) < 2:
            seeds, seed_safe = X_min, safe
        nn = NearestNeighbors(n_neighbors=min(k, len(seeds) - 1) + 1).fit(seeds)
        _, nbrs = nn.kneighbors(seeds); nbrs = nbrs[:, 1:]
        out = []
        for i in range(n_target):
            si = rng.integers(0, len(seeds))
            ni = nbrs[si, rng.integers(0, nbrs.shape[1])]
            lam = rng.random() * (0.5 if not seed_safe[si] else 1.0)
            out.append(seeds[si] + lam * (seeds[ni] - seeds[si]))
        self.diagnostics_ = {"rsmote_noise_seeds": int(noise.sum())}
        return (np.vstack([X, np.array(out)]),
                np.concatenate([y, np.full(n_target, minc)]))


class GKSMOTEApprox:
    def __init__(self, random_state=0, sampling_ratio=1.0):
        self.random_state = random_state
        self.sampling_ratio = sampling_ratio
        self.diagnostics_ = {}

    @staticmethod
    def _density(P, Q, bw):
        # Gaussian KDE of points P evaluated against reference set Q (chunked)
        d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2 * bw ** 2)).mean(axis=1)

    def fit_resample(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        maj, minc, n_maj, n_min = _split_classes(X, y)
        n_target = int(round(self.sampling_ratio * n_maj)) - n_min
        if n_target <= 0:
            return X, y
        X_min, X_maj = X[y == minc], X[y == maj]
        d = X.shape[1]
        bw = max(1.06 * X.std() * (n_min ** (-1.0 / (d + 4))), 1e-3)
        dens_min = self._density(X_min, X_min, bw)
        w = dens_min / dens_min.sum()
        accepted, attempts = [], 0
        while len(accepted) < n_target and attempts < 20 * n_target:
            take = min(512, (n_target - len(accepted)) * 2)
            parents = X_min[rng.choice(n_min, size=take, p=w)]
            cand = parents + rng.normal(0, bw * 0.5, size=parents.shape)
            dm = self._density(cand, X_min, bw)
            dM = self._density(cand, X_maj, bw)
            keep = cand[dm >= dM]
            accepted.extend(keep.tolist()); attempts += take
        if len(accepted) < n_target:
            short = n_target - len(accepted)
            accepted.extend(X_min[rng.integers(0, n_min, size=short)].tolist())
        self.diagnostics_ = {"gk_attempts": int(attempts)}
        return (np.vstack([X, np.array(accepted[:n_target])]),
                np.concatenate([y, np.full(n_target, minc)]))
