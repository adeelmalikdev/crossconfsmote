# CrossConfSMOTE

**Trust, but Verify: Cross-Conformal Auditing of Oversampling under Minority-Class Label Contamination**

Code, results and manuscript for a paper that asks a simple question: when some of your minority-class labels are wrong, which training points should an oversampler be allowed to copy from?

Status: under review at TMLR (submission 14407, submitted 12 November 2026). This repository is private while the paper is under review.

## The idea

Oversamplers like SMOTE generate synthetic minority points by interpolating between real minority points, called seeds. If a seed is actually a majority point wearing a wrong minority label, SMOTE happily builds synthetic points around it and the contamination spreads. Most pipelines never check their seeds.

CrossConfSMOTE puts a gate in front of generation:

1. A cross-conformal Mondrian audit scores every minority seed. The audit is class-conditional (Mondrian) and cross-fitted, so every seed is judged by models that never trained on it.
2. A seed is trusted only if its conformal prediction set is exactly {minority}. Anything else is suspicious.
3. Suspicious seeds are excluded from generation only. They stay in the training data as labelled points; they are just never used as interpolation parents.

That is the whole method. The claim is not a new sampler. The claim is that auditing seeds with conformal prediction gives you a much cleaner trusted set than the obvious alternative, a plain probability threshold, at the same downstream accuracy.

## Headline results

Benchmark: 8 datasets x 5 contamination conditions x 2 classifiers, with known ground-truth contamination masks so gate quality is measured, not assumed.

- Best mean PR-AUC across the benchmark: **0.5716**.
- Holm-corrected significant wins over SMOTE, Borderline-SMOTE, ADASYN, SMOTEENN, class weighting and no resampling.
- Gate diagnostic, fake-seed recall: the cross-conformal gate catches **0.89 to 0.92** of the planted fake seeds. A probability threshold catches **0.30 to 0.37**.
- Trusted-set contamination at 0.5x / 1x / 2x contamination: **0.05 / 0.13 / 0.21** for the conformal gate, against **0.28 / 0.42 / 0.57** for the probability gate.
- True-seed retention: **0.78 to 0.84** for the conformal gate, against about **0.89+** for the probability gate. The conformal gate is stricter and throws away a few more good seeds. That is the price, and the paper reports it openly.

One honest limit, stated in the paper and repeated here: downstream accuracy is statistically tied with the probability-threshold variant (ProbSMOTE). The difference is +0.002 with p = 0.37, which is a tie, not a win. What the conformal gate buys is a dramatically cleaner trusted seed set at the same accuracy, not better accuracy.

## Known failed cell

ADASYN fails on credit-g at 0.5x contamination. The cell is reported as a failed cell in the results files. It is not patched over and not hidden.

## Repository layout

- `code/` - the experiment code.
  - `crossconf.py` - CrossConfSMOTE itself.
  - `confsmote.py` - the conformal machinery, ProbSMOTE and the Mondrian conformal predictor.
  - `common.py` - shared utilities: contamination injection with known ground-truth masks, and metrics.
  - `run_experiment.py` - the main benchmark runner.
  - `diag_gate.py` - the gate diagnostic (fake-seed recall, trusted-set contamination, retention).
  - `head2head.py`, `extra_baselines.py`, `analyse.py`, `figures.py`, `top_up.py`, `top_up_yeast.py` - comparisons, extra baselines, analysis and figures.
- `results/` - the raw per-dataset CSVs, aggregated blocks, the gate diagnostic table and the Wilcoxon test results behind every number above.
- `paper/` - the submitted manuscript (PDF and LaTeX source), bibliography and figures.
- `SUBMISSION_CHECKLIST.md` - the pre-submission checklist used for the TMLR submission.

## Data

No data files are included. All eight datasets are public benchmarks fetched from OpenML by dataset ID at run time (see `OPENML_IDS` in `code/run_experiment.py`): mammography (310), yeast_ml8 (316), ozone_level_8hr (1487), credit-g (31), oil_spill (311), pc1, satellite and sick. Contamination is injected by the code with fixed seeds, with the ground-truth mask recorded, which is what makes the gate diagnostic possible.

## Running it

```bash
pip install -r requirements.txt
cd code
python run_experiment.py --config pilot --out ../results/pilot.csv
python diag_gate.py
```

Requirements: numpy, pandas, scikit-learn, imbalanced-learn, scipy, matplotlib.

## Citation

If you use this work, please cite the TMLR submission for now:

Muhammad Adeel Tariq. "Trust, but Verify: Cross-Conformal Auditing of Oversampling under Minority-Class Label Contamination." Submitted to Transactions on Machine Learning Research (TMLR), submission 14407, November 2026.

## License

MIT, see `LICENSE`. The datasets belong to their original providers and are fetched from OpenML under their own terms.
