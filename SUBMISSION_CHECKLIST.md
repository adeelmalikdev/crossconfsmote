# Submission package — CrossConfSMOTE paper
Prepared Oct 31, 2026. Status: **evidence complete, manuscript drafted. Nothing submitted anywhere.**

## What's in this folder
- `manuscript.tex` — full draft (abstract, method, theory, protocol, results, limitations). Compile on Overleaf (no local LaTeX on this machine). Figures referenced as `fig_ranks.png`, `fig_gate.png` (same folder).
- `fig_gate.png` — money figure: trusted-set contamination + fake recall vs noise level.
- `fig_ranks.png` — average ranks, all 13 methods.
- Raw evidence: `../results/` (per-fold CSVs, `aggregated_blocks.csv`, `aggregated_blocks_13.csv`, `wilcoxon_crossconf.csv`, `gate_diag.csv`). Code: `../code/` (crossconf.py is the method; everything seeded and rerunnable).

## Verified headline claims (do not strengthen these in editing)
1. Best average rank (3.79/10 primary methods); significant Holm-corrected wins over SMOTE, Borderline, ADASYN, SMOTEENN, class weights, no sampling, split-conformal v1.
2. Significant wins over GK-SMOTE / RSMOTE / SMOTE-IPF **reimplementations** (always keep that word).
3. Statistical TIE with ProbSMOTE downstream — the paper's claim is control/gate purity, NOT accuracy superiority over probability gating.
4. Gate: 89–92% fake recall; trusted-set contamination 4.8/12.5/20.6% at 0.5×/1×/2× vs 28/42/57% (probability gate).

## Before submitting — Adeel's actions (lead-time first)
- [ ] **ORCID iD** — register at orcid.org (effectively mandatory for IEEE; free, 10 min).
- [ ] **arXiv endorsement** — first-time submitters in cs.LG need an endorser or institutional-email route; arrange WEEKS before you want the preprint live. Target: preprint + public code in November 2026 so Fall 2027 applications can cite "preprint + under review".
- [ ] Decide on PeerJ APC ($2,155) — or take the zero-cost route first.
- [ ] Verify the two bracketed citations in manuscript.tex (RSMOTE 2021, GK-SMOTE author lists) before submission.
- [ ] Read the manuscript once yourself — your name is on it; every claim in it is checkable from the results folder.

## Venue sequence (never dual-submit; sequential only)
1. **arXiv + GitHub** (Nov 2026) — visibility for applications.
2. **PeerJ Computer Science** — soundness-only review fits an incremental-but-proven paper; ~35-day first decision; APC $2,155.
3. If rejected → **IJCNN 2027** (deadline 31 Jan 2027; chairs target ~40% acceptance; note: no-show = removed from Xplore, budget the trip before choosing this).
4. Standing zero-cost fallback: **Applied Intelligence** (Springer, subscription route $0).

## Honest odds (from the venue research)
No legitimate venue offers 90%. Realistic per-attempt odds at the venues above: ~28–40%. The sequencing above is what maximises the cumulative chance; the evidence package is built so a rejection at one venue needs no new experiments at the next.
