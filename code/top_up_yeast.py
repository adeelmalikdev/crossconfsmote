"""Top up the single missing (minority_loss, rep=1) cell for yeast_ml8."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import run_experiment as R

cfg = R.CONFIGS["full"]
X, y = R.load_dataset("yeast_ml8")
nk, nl, rep = "minority_loss", 0.2, 1
seed = 1000 + rep * 77 + int(nl * 100)
rows = R.run_one(X, y, nk, nl, rep, seed, cfg["methods"], cfg["classifiers"])
for r in rows:
    r["dataset"] = "yeast_ml8"
out = "../results/full_yeast_ml8.csv"
pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                          header=not os.path.exists(out))
print("TOPPED UP", len(rows))
