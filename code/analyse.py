"""
Benchmark statistics (Demsar-style, done honestly):
  * Unit of analysis = one (dataset x condition x classifier) block;
    scores averaged over folds and repeats first (no pseudoreplication).
  * Average ranks, Friedman omnibus on complete blocks.
  * Wilcoxon signed-rank of CrossConfSMOTE vs each rival across blocks,
    Holm-corrected, with mean delta and win/tie/loss counts.
"""
import glob
import numpy as np, pandas as pd
from scipy import stats

files = sorted(glob.glob("../results/full_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"rows={len(df)} datasets={sorted(df.dataset.unique())}")

agg = (df.groupby(["dataset", "noise_kind", "noise_level", "classifier", "method"])
         [["pr_auc", "mcc"]].mean().reset_index())
agg["block"] = (agg.dataset + "|" + agg.noise_kind + "|" +
                agg.noise_level.astype(str) + "|" + agg.classifier)

piv = agg.pivot_table(index="block", columns="method", values="pr_auc")
methods = list(piv.columns)
ranks = piv.rank(ascending=False, axis=1)
print("\n== AVERAGE RANKS (PR-AUC, lower=better), all blocks ==")
print(ranks.mean().sort_values().round(2).to_string())
print("\n== MEAN PR-AUC ==")
print(piv.mean().sort_values(ascending=False).round(4).to_string())

complete = piv.dropna()
print(f"\nFriedman on {len(complete)} complete blocks: "
      f"chi2={stats.friedmanchisquare(*[complete[m] for m in methods]).statistic:.1f} "
      f"p={stats.friedmanchisquare(*[complete[m] for m in methods]).pvalue:.2e}")

print("\n== CrossConfSMOTE vs rivals (Wilcoxon over all blocks, paired where both present) ==")
rows = []
for m in methods:
    if m in ("crossconf",):
        continue
    pair = piv[["crossconf", m]].dropna()
    d = pair["crossconf"] - pair[m]
    if (d != 0).sum() > 5:
        w = stats.wilcoxon(pair["crossconf"], pair[m])
        p = w.pvalue
    else:
        p = np.nan
    rows.append(dict(rival=m, blocks=len(pair), mean_delta=d.mean(),
                     wins=int((d > 1e-9).sum()), ties=int((abs(d) <= 1e-9).sum()),
                     losses=int((d < -1e-9).sum()), p_raw=p))
res = pd.DataFrame(rows)
# Holm
ps = res["p_raw"].values; order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for i, idx in enumerate(order):
    run = max(run, ps[idx] * (len(ps) - i)); adj[idx] = min(run, 1.0)
res["p_holm"] = adj
print(res.sort_values("mean_delta", ascending=False).round(4).to_string(index=False))

print("\n== per-condition average ranks (PR-AUC) ==")
for (nk, nl), sub in agg.groupby(["noise_kind", "noise_level"]):
    p2 = sub.pivot_table(index="block", columns="method", values="pr_auc")
    r = p2.rank(ascending=False, axis=1).mean().sort_values()
    print(f"{nk}@{nl}: " + ", ".join(f"{m}={v:.2f}" for m, v in r.head(4).items())
          + f" | crossconf mean={p2['crossconf'].mean():.4f} "
            f"best-other={p2.drop(columns=['crossconf']).mean().max():.4f}")

print("\n== MCC@0.5 average ranks ==")
pivm = agg.pivot_table(index="block", columns="method", values="mcc")
print(pivm.rank(ascending=False, axis=1).mean().sort_values().round(2).to_string())
pair = pivm[["crossconf", "none"]].dropna()
print(f"MCC crossconf vs none: mean delta {(pair['crossconf']-pair['none']).mean():+.4f}, "
      f"wins {int((pair['crossconf']>pair['none']).sum())}/{len(pair)}")

agg.to_csv("../results/aggregated_blocks.csv", index=False)
res.to_csv("../results/wilcoxon_crossconf.csv", index=False)
print("\nsaved aggregated_blocks.csv + wilcoxon_crossconf.csv")
