"""
Experiment runner: one (dataset x noise-condition x repeat) grid, samplers
applied INSIDE the training fold only, test set untouched at original
prevalence. One row per (config x method x classifier) in a CSV.

Usage:
  python run_experiment.py --config pilot --out ../results/pilot.csv
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from confsmote import ConfSMOTE, ProbSMOTE
from crossconf import CrossConfSMOTE
from extra_baselines import SMOTEIPF, RSMOTEApprox, GKSMOTEApprox
from common import inject_contamination, inject_minority_loss, score_all


# --------------------------------------------------------------------------- #
def get_classifiers():
    return {
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              random_state=0),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        "lr": LogisticRegression(max_iter=2000, class_weight=None),
    }


def get_samplers(seed):
    """name -> callable(X, y) -> (Xr, yr, diagnostics dict)"""
    def wrap(sampler):
        def f(X, y):
            Xr, yr = sampler.fit_resample(X, y)
            return Xr, yr, {}
        return f

    def conf(**kw):
        def f(X, y):
            s = ConfSMOTE(random_state=seed, **kw)
            Xr, yr = s.fit_resample(X, y)
            return Xr, yr, dict(s.diagnostics_)
        return f

    def prob(**kw):
        def f(X, y):
            s = ProbSMOTE(random_state=seed, **kw)
            Xr, yr = s.fit_resample(X, y)
            return Xr, yr, dict(s.diagnostics_)
        return f

    def extra(cls):
        def f(X, y):
            s = cls(random_state=seed)
            Xr, yr = s.fit_resample(X, y)
            return Xr, yr, dict(s.diagnostics_)
        return f

    def xconf(**kw):
        def f(X, y):
            s = CrossConfSMOTE(random_state=seed, **kw)
            Xr, yr = s.fit_resample(X, y)
            return Xr, yr, dict(s.diagnostics_)
        return f

    def none(X, y):
        return X, y, {}

    def classweight(X, y):   # marker: handled at classifier level below
        return X, y, {}

    return {
        "none": none,
        "class_weight": classweight,
        "ros": wrap(RandomOverSampler(random_state=seed)),
        "smote": wrap(SMOTE(random_state=seed, k_neighbors=5)),
        "borderline": wrap(BorderlineSMOTE(random_state=seed, k_neighbors=5)),
        "adasyn": wrap(ADASYN(random_state=seed, n_neighbors=5)),
        "smoteenn": wrap(SMOTEENN(random_state=seed)),
        "smotetomek": wrap(SMOTETomek(random_state=seed)),
        "confsmote": conf(),
        "confsmote_seedonly": conf(audit_synthetics=False),
        "confsmote_synonly": conf(audit_seeds=False),
        "probsmote": prob(),
        "crossconf": xconf(),
        "crossconf_seedonly": xconf(audit_synthetics=False),
        "crossconf_synonly": xconf(audit_seeds=False),
        "smoteipf": extra(SMOTEIPF),
        "rsmote": extra(RSMOTEApprox),
        "gksmote": extra(GKSMOTEApprox),
    }


CLASS_WEIGHT_METHODS = {"class_weight"}


# --------------------------------------------------------------------------- #
OPENML_IDS = {"mammography": 310, "yeast_ml8": 316, "ozone_level_8hr": 1487,
              "credit-g": 31, "creditcard": 1597, "oil_spill": 311,
              "sick": 38, "pc1": 1068, "satellite": 40900}

def load_dataset(name):
    from sklearn.datasets import fetch_openml
    try:
        ds = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, parser="auto")
        X = pd.get_dummies(ds.data).astype(float).to_numpy()
        y_raw = np.asarray(ds.target)
    except Exception:
        # pandas-arff parser chokes on some OpenML files (int categories);
        # fall back to the raw numpy path and coerce features to numeric.
        ds = fetch_openml(data_id=OPENML_IDS[name], as_frame=False, parser="liac-arff")
        Xdf = pd.DataFrame(ds.data)
        X = Xdf.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
        y_raw = np.asarray(ds.target)
    vals, cnts = np.unique(y_raw, return_counts=True)
    minority_val = vals[np.argmin(cnts)]
    y = (y_raw == minority_val).astype(int)
    return np.asarray(X, dtype=float), y


# --------------------------------------------------------------------------- #
def run_one(X, y, noise_kind, noise_level, repeat, seed, methods, classifiers):
    rng = np.random.default_rng(seed)
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr_clean, y_te = y[tr], y[te]
        # median imputation fitted on TRAIN only (some OpenML sets, e.g.
        # sick, carry missing values that neighbour-based samplers reject)
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        ii = np.where(np.isnan(X_tr)); X_tr = X_tr.copy(); X_tr[ii] = np.take(med, ii[1])
        ii = np.where(np.isnan(X_te)); X_te = X_te.copy(); X_te[ii] = np.take(med, ii[1])
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        # --- inject noise into TRAIN ONLY; keep the fake-seed mask -------- #
        fake_mask = np.zeros(len(y_tr_clean), dtype=bool)
        if noise_kind == "contamination":
            n_min = int(y_tr_clean.sum())
            n_fake = int(round(noise_level * n_min))
            y_tr, fake_mask = inject_contamination(X_tr_s, y_tr_clean, n_fake, rng)
        elif noise_kind == "minority_loss":
            y_tr = inject_minority_loss(y_tr_clean, noise_level, rng)
        else:
            y_tr = y_tr_clean

        samplers = get_samplers(seed + fold)
        for mname in methods:
            fn = samplers[mname]
            t0 = time.time()
            try:
                Xr, yr, diag = fn(X_tr_s, y_tr)
            except Exception as e:
                print(f"  !! sampler {mname} failed: {e}")
                continue
            dt = time.time() - t0

            # gate-quality diagnostic for ConfSMOTE variants: how many of the
            # INJECTED fake seeds were excluded from generation is measured in
            # a dedicated diagnostic run (see diag_gate.py); here we record
            # the sampler's own diagnostics.
            for cname in classifiers:
                clf = get_classifiers()[cname]
                if mname in CLASS_WEIGHT_METHODS:
                    if cname == "hgb":
                        # HGB has no class_weight; use sample weights instead
                        w = np.where(yr == 1, len(yr) / (2 * yr.sum()), 0.5)
                        clf.fit(Xr, yr, sample_weight=w)
                    elif cname == "rf":
                        clf.set_params(class_weight="balanced")
                        clf.fit(Xr, yr)
                    else:
                        clf.set_params(class_weight="balanced")
                        clf.fit(Xr, yr)
                else:
                    clf.fit(Xr, yr)
                proba = clf.predict_proba(X_te_s)[:, list(clf.classes_).index(1)]
                m = score_all(y_te, proba)
                m.update(dict(noise_kind=noise_kind, noise_level=noise_level,
                              repeat=repeat, fold=fold, method=mname,
                              classifier=cname, prevalence=float(np.mean(y_te)),
                              runtime_s=dt,
                              seed_reject_rate=diag.get("seed_reject_rate", np.nan),
                              syn_accept_rate=diag.get("syn_accept_rate", np.nan)))
                rows.append(m)
    return rows


# --------------------------------------------------------------------------- #
CONFIGS = {
    "pilot_fast": dict(
        datasets=["mammography", "yeast_ml8", "ozone_level_8hr", "credit-g"],
        conditions=[("contamination", 0.0), ("contamination", 1.0),
                    ("contamination", 2.0), ("minority_loss", 0.2)],
        repeats=1,
        methods=["none", "class_weight", "smote", "smoteenn",
                 "confsmote", "confsmote_seedonly", "confsmote_synonly",
                 "probsmote"],
        classifiers=["hgb"],
    ),
    "pilot_rest": dict(
        datasets=["yeast_ml8", "ozone_level_8hr", "credit-g"],
        conditions=[("contamination", 0.0), ("contamination", 1.0),
                    ("contamination", 2.0), ("minority_loss", 0.2)],
        repeats=1,
        methods=["none", "class_weight", "smote", "smoteenn",
                 "confsmote", "confsmote_seedonly", "confsmote_synonly",
                 "probsmote"],
        classifiers=["hgb"],
    ),
    "full": dict(
        datasets=None,  # set per-run via --datasets
        conditions=[("contamination", 0.0), ("contamination", 0.5),
                    ("contamination", 1.0), ("contamination", 2.0),
                    ("minority_loss", 0.2)],
        repeats=2,
        methods=["none", "class_weight", "smote", "borderline", "adasyn",
                 "smoteenn", "probsmote", "confsmote", "crossconf",
                 "crossconf_seedonly"],
        classifiers=["hgb", "rf"],
    ),
    "extra": dict(
        datasets=None,  # set per-run via --datasets
        conditions=[("contamination", 0.0), ("contamination", 0.5),
                    ("contamination", 1.0), ("contamination", 2.0),
                    ("minority_loss", 0.2)],
        repeats=2,
        methods=["smoteipf", "rsmote", "gksmote"],
        classifiers=["hgb", "rf"],
    ),
    "pilot": dict(
        datasets=["mammography", "yeast_ml8", "ozone_level_8hr", "credit-g"],
        conditions=[("contamination", 0.0), ("contamination", 0.5),
                    ("contamination", 1.0), ("contamination", 2.0),
                    ("minority_loss", 0.2)],
        repeats=2,
        methods=["none", "class_weight", "smote", "borderline", "adasyn",
                 "smoteenn", "confsmote", "confsmote_seedonly",
                 "confsmote_synonly", "probsmote"],
        classifiers=["hgb", "rf", "lr"],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset names overriding the config")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    if args.datasets:
        cfg = dict(cfg); cfg["datasets"] = args.datasets.split(",")
    import os
    total = 0
    for dname in cfg["datasets"]:
        print(f"== dataset {dname}", flush=True)
        X, y = load_dataset(dname)
        print(f"   n={len(y)} prevalence={np.mean(y):.4f} d={X.shape[1]}", flush=True)
        for (nk, nl) in cfg["conditions"]:
            for rep in range(cfg["repeats"]):
                seed = 1000 + rep * 77 + int(nl * 100)
                rows = run_one(X, y, nk, nl, rep, seed,
                               cfg["methods"], cfg["classifiers"])
                for r in rows:
                    r["dataset"] = dname
                # incremental append: partial progress survives interruptions
                pd.DataFrame(rows).to_csv(
                    args.out, mode="a", index=False,
                    header=not os.path.exists(args.out))
                total += len(rows)
                print(f"   {nk}@{nl} rep{rep}: {len(rows)} rows", flush=True)
    print(f"WROTE {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
