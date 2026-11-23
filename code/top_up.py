"""Run only the (condition, rep) blocks missing from a dataset's CSV."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import run_experiment as R

dname = sys.argv[1]
cfgname = sys.argv[2] if len(sys.argv) > 2 else "full"
cfg = R.CONFIGS[cfgname]
prefix = "full" if cfgname == "full" else "extra"
out = f"../results/{prefix}_{dname}.csv"
have = set()
if os.path.exists(out):
    d0 = pd.read_csv(out)
    have = set(map(tuple, d0[["noise_kind", "noise_level", "repeat"]]
                   .drop_duplicates().values.tolist()))
X, y = R.load_dataset(dname)
for (nk, nl) in cfg["conditions"]:
    for rep in range(cfg["repeats"]):
        if (nk, float(nl), rep) in have:
            continue
        seed = 1000 + rep * 77 + int(nl * 100)
        rows = R.run_one(X, y, nk, nl, rep, seed,
                         cfg["methods"], cfg["classifiers"])
        for r in rows:
            r["dataset"] = dname
        pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                  header=not os.path.exists(out))
        print(f"topped up {nk}@{nl} rep{rep}: {len(rows)} rows", flush=True)
print("TOPUP_DONE", dname)
