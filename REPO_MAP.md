# Repository map: `paper1_reproduction`

Reproduction bundle and revision work for *"What Does a Game-Trained Embedding Learn? Strategy
Ensembles versus Latent Overhead in Pursuit–Evasion"* (Jimenez et al., draft of 8 Oct 2026).
Pursuit–evasion on the flat torus, open-loop multi-head actor trained by differentiable
self-play, and the analysis of what its latent space encodes.

Three layers, in the order they were made:

| layer | when | where | what |
|---|---|---|---|
| **Original bundle** | 7 Oct 2026 | repo root, `paper/`, `revision/` | the code, checkpoints, results and figures behind the 8 Oct PDF |
| **Review and revision** | 8–26 Oct 2026 | `REVIEW_claude.md`, `REVISION_PLAN.md`, `New_Revision_plan.md`, `rev2/` | independent review of the draft, corrected metrics, a 120-run baseline ensemble (R1), a summary PDF |
| **Orientation and minimal game** | 28 Oct 2026 | `viz/` | trajectory/latent visualiser, the simplest exactly zero-sum game as a pilot baseline, orientation notes |

No file of the original bundle has been modified by the later layers. Everything new lives in
`rev2/` and `viz/` plus the four Markdown files at the root.

---

## 1. Original bundle (7 Oct 2026)

### Core model code

| file | what it does |
|---|---|
| `pursuit_evasion_numpy.py` | **The actor, end to end**, in pure NumPy. Fourier features of time → one tanh layer (16-dim latent H(t)) → 16 linear heads → unit headings; explicit midpoint rollout of both players on the torus with wrap; soft capture-time loss with no-capture penalty and distance shaping; hand-written backward pass; Adam; `train()`; basic plotters. Not a PINN: the ODE is integrated inside the loss. |
| `pursuit_evasion_v2.py` | The **critic**: value net V(Δ) on the torus trained on the HJI residual (the only PINN in the repo); closed-form `analytic_V`; capture-time map; T^d demo. |
| `pursuit_evasion_v3.py`, `pursuit_evasion_v4.py` | Variants used by v3/v4/v5 experiments (heads sweep, distillation, obstacle; "honest" critic with no analytic input). |
| `geometries.py` | Ball, disk and Möbius geometries for the critic study. |

### Experiment drivers (one script per arm of the paper)

| paper item | driver | outputs in root / `revision/` |
|---|---|---|
| §2 baseline training | `run_v2.py` (warm-up 1500, 14k iters at 3:1 steps, Adam 1e-2/1e-3 reset every 3500, λ_d = 5, λ_nc = 2) | `v2_actor.npz` (published actor), `v2_V.npz`, `v2_V3d.npz`, `v2_summary.json` |
| §3 PR of latent vs strategies | `embedding_analysis.py`, `oracle_pr.py`, `pr_capturable_scrutiny.py`, `revision/post_hoc_checks.py` | `embedding_summary.json`, `pedro_oracle_pr_result.json`, `post_hoc_checks*.json` |
| Table 1 C1 seeds / IC resamples | `run_c1.py`, `run_c1_ic.py` | `c1_s1..7.npz`, `c1_ic1..5.npz`, `revision/c1_seeds.json` |
| Table 1 C2 symmetrised exact zero-sum | `run_c2.py`, `run_c2_seeds.py` | `c2_*.npz` (snapshots at 20k–28k, 7 seeds), `revision/c2_*.json` |
| Table 1 C3 bandwidth sweep (Fig. 1) | `run_c3.py` | `c3_k6_actor.npz`, `c3_k24_actor.npz`, `revision/c3_break_analysis.json` |
| Table 1 C4 sector, C5 shaping sweep | `run_c4c5.py` | `c4_actor.npz`, `c5_lam{0,1,5}_actor.npz`, `revision/c4c5_analysis.json` |
| Table 1 C7 warm-start collapse | `run_c7.py` | `c7_fit.npz`, `c7_actor.npz`, `revision/c2c7_analysis.json` |
| §5 winding / sectors (Figs. 2–3) | `run_e17.py`, `run_e1b_reflected.py`, `winding_analysis.py`, `make_fig_sectors.py`, `make_fig_winding_reflection.py`, `revision/analyze_e1e7.py` | `e1_actor.npz`, `e1b_*.npz`, `e7_*.npz`, `pedro_winding_result.json`, `revision/e1e7_analysis.json`, `revision/winding_truncated.json` |
| §6 Δv sweep (Fig. 4) | `run_dv_sweep.py`, `analyze_dv_sweep.py` | `dv_sweep_results_full.jsonl`, `dv_sweep_summary.json` |
| §8 protocols, honest corrections, Shapley (Figs. 6–8) | `run_game_protocols.py`, `run_honest_protocols.py`, `shap_attribution.py`, `analyze_game_protocols.py`, `analyze_honest_protocols.py`, `analyze_shap.py` | `game_protocols_*`, `honest_protocols_*`, `shap_seed{0,1,2}.json` |
| §9 critic and geometries | `run_v4.py`, `run_v5.py`, `run_geom.py`, `run_ext.py`, `honest_critic_geom.py`, `revision/post_hoc_checks2.py` | `v4_honestV.npz`, `v5_*.npz`, `geom_*.npz`, `geom_summary.json`, `v4/v5_summary.json` |
| not in the PDF | `run_v3.py` (heads sweep, distillation, obstacle), `run_train.py` (smoke test) | `v3_*.npz`, `v3_summary.json`, `checkpoint.npz` |

`README_REPRODUCE.md` is the original authors' run order. `paper/` holds the draft
(`embeddings_paper.tex/.pdf`, `jmlr2e.sty`) and its 24 figures. `revision/` holds the original
authors' post-hoc checks and analysis JSONs (same date as the bundle).

---

## 2. Review and revision (8–26 Oct 2026)

### Documents at the root

| file | date | content |
|---|---|---|
| `REVIEW_claude.md` | 8 Oct | Independent review of the draft against the bundle. Every number recomputed. Lists what holds and what does not. |
| `REVISION_PLAN.md` | 22 Oct | Full revision document: assessment of the review, five further problems (PR(Z) is gauge-dependent; training is chaotic so seeds are not identifiers; arms mixed protocols; baseline IC set is an outlier; "mediocre equilibrium" mis-measured), the run plan R1–R11, and **§4: the R1 results** (census tables, pre/post-capture constancy, chart analysis, closed-form verification). |
| `New_Revision_plan.md` | 22 Oct | Same document with §4 removed: the plan only, R1 marked done, R2 (capture-truncated loss) as the next priority. |

### `rev2/` — corrected metrics and the R1 ensemble

| file | what it does |
|---|---|
| `core.py` | `RunConfig` dataclass holding every baseline hyper-parameter; `train_actor()` runs the published protocol deterministically (seeded init, seeded ICs, optional Adam reset); checkpoint loader. Wraps the original `pursuit_evasion_numpy.py` functions, does not change them. |
| `metrics.py` | Gauge-invariant replacements for the paper's statistics: Gram PR across heads of the unit-heading curves (the object bounded by Proposition 1), constancy η, temporal PR of û and of z, PR(H) centred and uncentred, chart choice vs optimal chart, capture time vs closed-form V*, static alignment with the saddle direction, analytic saddle reference. |
| `test_metrics.py` | Gate: reproduces all 13 legacy numbers of the draft from `v2_actor.npz` before any new number is trusted. |
| `legacy_checkpoints.py` | Re-scores every published checkpoint with the corrected metrics → `results/legacy_corrected.json`. |
| `run_r1.py` | R1: 120 runs under one protocol (10 init seeds × 10 IC seeds, plus a 10-run chaos arm at 1e-10 perturbation and a 10-run no-reset arm). ~1 h. |
| `analyze_r1.py` | Produces every R1 number and figure: `results/r1_summary.json`, `results/r1_census.md`, `figs/r1_*.{png,pdf}` (dimensionalities, capture windows, charts, dynamics, game, spectra, variance). |
| `ckpt/r1/` (120 `.npz`), `results/r1/` (120 `.json`), `logs/` | The R1 checkpoints, per-run metrics and run logs. |
| `summary/paper1_summary.tex/.pdf` | 26 Oct: a stand-alone summary of the paper's claims, the evidence, and the review verdict section by section, prepared for the coauthors. |

Headline R1 findings (details in `REVISION_PLAN.md` §4 and `rev2/results/r1_census.md`):
pre-capture headings are constant to four decimals and sit on the saddle of one image chart;
all excess over rank 2 is post-capture motion driven by the shaping term; the chart choice is
the game's real discrete content; the published baseline is an outlier of the ensemble.

---

## 3. Orientation and the minimal game (28 Oct 2026)

### `viz/`

| file | what it does |
|---|---|
| `NOTES_orientation.md` | Written explanation of the network (inputs, outputs, loss, not a PINN), a code map per paper section, the paper's vocabulary (actor, oracle, winding, open loop, chart, …), what the runs look like, PR vs PCA, S vs H, T^d, and a review of the baseline with a proposed ladder L0/L1/L2 of simpler baselines. |
| `visualize_actor.py` | Loads any actor checkpoint (legacy or rev2 layout), re-rolls the game, and draws: the 16 head trajectories on the fundamental domain with saddle direction, reach circle and capture mark; the same paths on the universal cover; the embedded 3-D torus; heading angle θ(t) against the saddle angle together with d(t); the latent curve H(t), its PCA profiles and spectrum; a per-head table of constancy before/after capture. `python3 viz/visualize_actor.py [ckpt] [out_dir]`. |
| `minimal_game_pilot.py` | The **simplest exactly zero-sum game** with the same architecture: single loss J = ⟨d_T⟩_t (or terminal d_T with `--terminal`), L_P = J, L_E = −J, no capture mechanics, no warm-up, 1:1 alternating steps at equal learning rate, finite-difference gradient check at start. Records Gram PR, constancy, PR(H) every 250 iterations, compares with the analytic open-loop saddle, saves actor and figures. |
| `baseline/` | The five figures and head table for the published actor `v2_actor.npz`. |
| `pilot_integ_tmax1.5_s0/` | Pilot, integrated distance, horizon 1.5 (no head reached): Gram PR 2.0, η 0.99, PR(H) ≈ 7. |
| `pilot_integ_tmax4_s0/` | Pilot, horizon 4 (six heads reached, pursuer weaves): Gram PR 4.3 / 3.4, η 0.6–0.7. |
| `pilot_term_tmax1.5_s0/` | Pilot, terminal distance, horizon 1.5. |
| `pilot_*.log` | Console logs of the three pilot runs. |

Pilot conclusion: the rank-2 strategy ensemble appears already in the simplest zero-sum game, with
no capture term, no shaping, no warm-up and no 3:1 steps; the latent overhead PR(H) ≈ 7 persists.
Letting the faster pursuer reach a head under a non-terminating fixed-speed rollout is what breaks
constancy, which is also the mechanism behind the post-capture excess in the published baseline
(the shaping term ⟨d⟩ weights post-capture time steps at full weight; see R2 in the plan).

---

## Running things

```bash
python3 rev2/test_metrics.py            # gate: reproduce the 13 legacy numbers
python3 rev2/run_r1.py                  # 120-run ensemble, ~1 h
python3 rev2/analyze_r1.py              # census, summary, figures
python3 viz/visualize_actor.py          # figures for v2_actor.npz -> viz/baseline/
python3 viz/minimal_game_pilot.py --tmax 1.5 --iters 3000
```

Dependencies: NumPy, SciPy, Matplotlib (Python 3.10 was used).
