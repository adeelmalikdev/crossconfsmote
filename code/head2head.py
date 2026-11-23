"""Head-to-head: ConfSMOTE v1 (split) vs CrossConfSMOTE v2 vs controls."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

import run_experiment as R
from confsmote import ConfSMOTE, ProbSMOTE
from crossconf import CrossConfSMOTE
from common import inject_contamination, score_all

METHODS = {
    "none": None,
    "smote": "smote",
    "probsmote": "prob",
    "confsmote_v1": "v1",
    "crossconf_v2": "v2",
    "crossconf_v2_seedonly": "v2s",
    "crossconf_v2_synonly": "v2y",
}

def apply(name, X, y, seed):
    tag = METHODS[name]
    if tag is None: return X, y, {}
    if tag == "smote":
        Xr, yr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y); return Xr, yr, {}
    if tag == "prob":
        s = ProbSMOTE(random_state=seed); Xr, yr = s.fit_resample(X, y); return Xr, yr, s.diagnostics_
    if tag == "v1":
        s = ConfSMOTE(random_state=seed); Xr, yr = s.fit_resample(X, y); return Xr, yr, s.diagnostics_
    if tag == "v2":
        s = CrossConfSMOTE(random_state=seed); Xr, yr = s.fit_resample(X, y); return Xr, yr, s.diagnostics_
    if tag == "v2s":
        s = CrossConfSMOTE(random_state=seed, audit_synthetics=False); Xr, yr = s.fit_resample(X, y); return Xr, yr, s.diagnostics_
    if tag == "v2y":
        s = CrossConfSMOTE(random_state=seed, audit_seeds=False); Xr, yr = s.fit_resample(X, y); return Xr, yr, s.diagnostics_

rows = []
for dname in ["mammography", "ozone_level_8hr"]:
    X, y = R.load_dataset(dname)
    print("==", dname, X.shape, flush=True)
    for nl in [0.0, 1.0, 2.0]:
        rng = np.random.default_rng(42)
        skf = StratifiedKFold(5, shuffle=True, random_state=7)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            ytr_n, _ = inject_contamination(Xtr, ytr, int(round(nl * ytr.sum())), rng)
            for mname in METHODS:
                Xr, yr, diag = apply(mname, Xtr, ytr_n, 100 + fold)
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
                clf.fit(Xr, yr)
                proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
                m = score_all(yte, proba)
                m.update(dict(dataset=dname, noise_level=nl, fold=fold, method=mname,
                              seed_reject_rate=diag.get("seed_reject_rate", np.nan)))
                rows.append(m)
        print("  done contamination", nl, flush=True)
pd.DataFrame(rows).to_csv("../results/head2head.csv", index=False)
print("WROTE", len(rows))
