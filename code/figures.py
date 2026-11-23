"""Paper figures from the actual result files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, glob

plt.rcParams.update({"font.size": 10, "figure.dpi": 200})
OUT = "../files/"

# ---- Fig 1: trusted-set contamination + fake recall (money figure) ------- #
g = pd.read_csv("../results/gate_diag.csv")
agg = g.groupby(["noise_level", "gate"])[["trusted_contam", "fake_recall",
                                           "true_retention"]].mean()
levels = [0.5, 1.0, 2.0]
styles = {"crossconf": ("o-", "CrossConfSMOTE (ours)"),
          "probsmote": ("s--", "Probability gate"),
          "confsmote": ("^--", "Split-conformal gate")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for gate, (st, lab) in styles.items():
    axes[0].plot(levels, [agg.loc[(l, gate), "trusted_contam"] for l in levels],
                 st, label=lab)
    axes[1].plot(levels, [agg.loc[(l, gate), "fake_recall"] for l in levels],
                 st, label=lab)
axes[0].axhline(0.10, color="grey", lw=0.8, ls=":")
axes[0].text(1.95, 0.115, "nominal $\\alpha=0.10$", ha="right", fontsize=8, color="grey")
axes[0].set_xlabel("Injected fakes (x true minority count)")
axes[0].set_ylabel("Contamination of trusted seed set")
axes[0].set_title("(a) Trusted-set contamination", fontsize=10)
axes[1].set_xlabel("Injected fakes (x true minority count)")
axes[1].set_ylabel("Injected fakes excluded from generation")
axes[1].set_title("(b) Fake-seed recall", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_gate.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: average ranks across 13 methods ------------------------------- #
main = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/full_*.csv"))], ignore_index=True)
extra = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/extra_*.csv"))], ignore_index=True)
df = pd.concat([main, extra], ignore_index=True)
a = df.groupby(["dataset", "noise_kind", "noise_level", "classifier", "method"])["pr_auc"].mean().reset_index()
a["block"] = a.dataset + "|" + a.noise_kind + "|" + a.noise_level.astype(str) + "|" + a.classifier
piv = a.pivot_table(index="block", columns="method", values="pr_auc")
r = piv.rank(ascending=False, axis=1).mean().sort_values()
names = {"crossconf": "CrossConfSMOTE (ours)", "probsmote": "ProbSMOTE",
         "crossconf_seedonly": "CrossConf (seed-only)", "confsmote": "ConfSMOTE (split)",
         "none": "No sampling", "gksmote": "GK-SMOTE*", "smote": "SMOTE",
         "rsmote": "RSMOTE*", "class_weight": "Class weights", "borderline": "Borderline-SMOTE",
         "adasyn": "ADASYN", "smoteenn": "SMOTEENN", "smoteipf": "SMOTE-IPF*"}
fig, ax = plt.subplots(figsize=(7.2, 3.4))
colors = ["#1a7f37" if m == "crossconf" else "#8b949e" if m == "probsmote" else "#57606a"
          for m in r.index]
ax.barh([names.get(m, m) for m in r.index][::-1], r.values[::-1], color=colors[::-1])
ax.set_xlabel("Average rank across 80 blocks (lower is better)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT + "fig_ranks.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
