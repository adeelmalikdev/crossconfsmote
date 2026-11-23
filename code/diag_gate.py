"""
Gate-quality diagnostics against the KNOWN injected fake-seed mask.

For each audit rule, on contaminated training folds, measure over the
minority-labelled pool (true seeds + injected fakes):
  * fake_recall       - fraction of injected fakes excluded from generation
  * true_retention    - fraction of true seeds kept for generation
  * trusted_contam    - fraction of KEPT seeds that are actually fake
                        (the quantity a conformal guarantee should control)
Rules: crossconf (v2), confsmote (v1 split-conformal), probsmote (raw prob).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import run_experiment as R
from common import inject_contamination
from confsmote import MondrianConformal
from crossconf import _pvals_from_cal

ALPHA = 0.10
SCORER = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                        random_state=0)

def gate_crossconf(X, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 2))
    for tr, ho in skf.split(X, y):
        m = clone(SCORER); m.fit(X[tr], y[tr]); oof[ho] = m.predict_proba(X[ho])
    cal = {c: 1.0 - oof[y == c, c] for c in (0, 1)}
    nc = {c: 1.0 - oof[:, c] for c in (0, 1)}
    p_min = _pvals_from_cal(cal[1], nc[1]); p_maj = _pvals_from_cal(cal[0], nc[0])
    return (p_min > ALPHA) & (p_maj <= ALPHA)   # trusted mask over ALL points

def _split_mc(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_i, cal_i = next(sss.split(X, y))
    return MondrianConformal(SCORER).fit(X[fit_i], y[fit_i],
                                         X[cal_i], y[cal_i])

def gate_confsmote(X, y, seed):
    conf = _split_mc(X, y, seed)
    pv = conf.p_values(X)
    return (pv[1] > ALPHA) & (pv[0] <= ALPHA)

def gate_probsmote(X, y, seed):
    mc = _split_mc(X, y, seed)
    proba = mc.clf_.predict_proba(X)
    ci = {c: i for i, c in enumerate(mc.classes_)}
    return (proba[:, ci[1]] > 0.5) & (proba[:, ci[0]] <= 0.5)

GATES = {"crossconf": gate_crossconf, "confsmote": gate_confsmote,
         "probsmote": gate_probsmote}

rows = []
for dname in ["mammography", "ozone_level_8hr", "satellite", "sick"]:
    X, y = R.load_dataset(dname)
    for nl in [0.5, 1.0, 2.0]:
        rng = np.random.default_rng(42)
        skf = StratifiedKFold(5, shuffle=True, random_state=7)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, ytr_clean = X[tr], y[tr]
            med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0, med)
            ii = np.where(np.isnan(Xtr)); Xtr = Xtr.copy(); Xtr[ii] = np.take(med, ii[1])
            sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr)
            ytr, fake_mask = inject_contamination(
                Xtr, ytr_clean, int(round(nl * ytr_clean.sum())), rng)
            pool = (ytr == 1)
            for gname, gfn in GATES.items():
                trusted = gfn(Xtr, ytr, 100 + fold) & pool
                kept_fake = int((trusted & fake_mask).sum())
                kept_true = int((trusted & ~fake_mask & pool).sum())
                rows.append(dict(
                    dataset=dname, noise_level=nl, fold=fold, gate=gname,
                    fake_recall=float(1 - kept_fake / max(fake_mask.sum(), 1)),
                    true_retention=float(kept_true / max((pool & ~fake_mask).sum(), 1)),
                    trusted_contam=float(kept_fake / max(trusted.sum(), 1)),
                    n_trusted=int(trusted.sum())))
        print(dname, nl, "done", flush=True)
        import os
        pd.DataFrame(rows).to_csv("../results/gate_diag.csv", mode="a",
                                  index=False,
                                  header=not os.path.exists("../results/gate_diag.csv"))
        rows = []
print("WROTE gate_diag.csv")
