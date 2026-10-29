# Paper 1 — "What Does a Game-Trained Embedding Learn?" — reproduction bundle

Code and data to reproduce the figures and numbers in
`paper/embeddings_paper.pdf`. Only Paper 1 code is included (all Paper 2 /
HJI-critic code has been removed). Run scripts **from the root of this bundle**;
all paths resolve relative to the bundle (no absolute paths).

## Environment
```
python >= 3.10
pip install numpy matplotlib        # enough for every figure/number below
pip install torch                   # ONLY if you want to re-train (regenerate the .npz)
```
You do **not** need torch to reproduce any figure or number: the trained outputs
(`*.npz`) and result files (`*.json`, `*.jsonl`) are included, and the analysis
scripts are pure numpy + matplotlib.

## What I verified
- All 37 scripts parse (syntax-clean).
- Every **numpy** analysis/figure script below was *executed* in a clean
  environment and reproduces its output; the regenerated
  `fig_dv_sweep/protocols/honest/shap.png` match the committed paper figures
  byte-for-byte.
- Bugs fixed while preparing this bundle (so they run out-of-the-box): the
  `analyze_*` scripts read from hard-coded `/tmp/*.jsonl` and wrote to a
  non-existent `paper1/` directory; the `revision/` scripts pointed at a dead
  session path. All repaired to resolve inside the bundle.
- The **torch** training scripts are syntax-checked only (no GPU here); they
  regenerate the `.npz` that the analyses consume — which are already included.

## Figures → script → command (reproduce from included data, seconds, CPU)

| figure | script / command | input |
|---|---|---|
| `fig_dv_sweep.png` (Δv sweep) | `python3 analyze_dv_sweep.py` | `dv_sweep_results_full.jsonl` |
| `fig_protocols.png` (game protocols) | `python3 analyze_game_protocols.py` | `game_protocols_results_full.jsonl` |
| `fig_honest.png` (honest protocols) | `python3 analyze_honest_protocols.py` | `honest_protocols_results.jsonl` |
| `fig_shap.png` (attribution) | `python3 analyze_shap.py` | `shap_seed{0,1,2}.json` |
| `fig_embed_head_gauge.png` (head singular values) | `python3 embedding_analysis.py` → `fig_embed_head_ortho.png` | `v2_actor.npz` |
| `fig_embed_intk_spectrum.png` (integer-k spectra) | `python3 embedding_analysis.py` → `fig_embed_fourier_shells.png` / `fig_embed_pca_spectrum.png` | `v2_actor.npz` |
| `fig_embed_sectors.png` (sector experiments) | `python3 make_fig_sectors.py` | `revision/e1e7_analysis.json`, `v2_actor.npz`, `e1b_actor.npz` |
| `fig_winding_reflection.png` (reflection control) | `python3 make_fig_winding_reflection.py` | `e1b_actor.npz`, `e1b_refl_actor.npz` |

`embedding_analysis.py` runs the full embedding decomposition from `v2_actor.npz`
and writes the panel set (`fig_embed_pca_spectrum`, `_head_ortho`,
`_fourier_shells`, `_pc_time_{E,P}`, `_headings_{E,P}`, `_stability`,
`_reconstruction`) plus `embedding_summary.json`. The paper's
`fig_embed_head_gauge` and `fig_embed_intk_spectrum` are polished/renamed
versions of these panels (same content).

## Numbers → script (Table 1 controls and the sector/winding results)

| paper element | numbers from | data file |
|---|---|---|
| **Tab. (controls) C1–C7** | `revision/post_hoc_checks.py`, `revision/analyze_e1e7.py` (training: `run_c1.py`, `run_c1_ic.py`, `run_c2.py`, `run_c2_seeds.py`, `run_c3.py`, `run_c4c5.py`, `run_c7.py`) | `revision/{c1_seeds,c2_seed_replicates,c2c7_analysis,c3_break_analysis,c4c5_analysis,within_run_ratios}.json`, `c*_*.npz` |
| **Sector experiment** (`fig_embed_sectors`) | `run_e17.py`, `run_c4c5.py`, `revision/analyze_e1e7.py` | `revision/e1e7_analysis.json`, `e7_*.npz` |
| **Reflection / winding** (`fig_winding_reflection`) | `run_e1b_reflected.py`, `winding_analysis.py`, `revision/analyze_e1e7.py` | `revision/winding_truncated.json`, `e1b_refl_actor.npz` |
| **PR(Z) / oracle-pursuer checks** | `oracle_pr.py`, `pr_capturable_scrutiny.py` | `pedro_oracle_pr_result.json`, `pedro_winding_result.json` |

Run e.g. `python3 revision/analyze_e1e7.py` (verified; writes
`e1e7_analysis.json`) or `python3 revision/post_hoc_checks.py`.

## All 8 figures reproduce (the last 2 reconstructed and verified)
`fig_embed_sectors.png` and `fig_winding_reflection.png` originally lacked a
committed plotter — their panels were rendered in a post-processing pass. They
have been reconstructed here as `make_fig_sectors.py` and
`make_fig_winding_reflection.py`, which recompute the per-head net winding
`w = dispE / L` straight from the actors (`v2_actor.npz`, `e1b_actor.npz`,
`e1b_refl_actor.npz`) with the exact pipeline of `revision/analyze_e1e7.py`, and
read the PR values from `revision/e1e7_analysis.json`. Both were checked against
the published panels and reproduce the same data: baseline has 12 winding
classes and E1b 3; the reflected cloud mirrors about `w_x = 0`
(mean `w_x`: +1.05 → −0.86).

## Training / data generation (optional; torch)
`pursuit_evasion_v2.py` / `v3.py` / `v4.py` are the embedding game trainers;
`run_v2.py … run_v5.py`, `run_c*.py`, `run_e*.py`, `run_dv_sweep.py`,
`run_game_protocols.py`, `run_honest_protocols.py`, `honest_critic_geom.py`,
`run_geom.py`, `geometries.py` regenerate the `.npz` / `.jsonl` consumed above.
`pursuit_evasion_numpy.py` is the pure-numpy simulator used by
`embedding_analysis.py`.

## Contents
- `*.py` — Paper 1 scripts (analysis + training).
- `revision/*.py`, `revision/*.json` — control/sector/winding analyzers and their
  precomputed results.
- `*.npz` (85), `*.jsonl`, `*.json` — trained actors, sweep results, summaries.
- `paper/` — the compiled paper, source (`embeddings_paper.tex`), `jmlr2e.sty`,
  and all figure PNGs.
