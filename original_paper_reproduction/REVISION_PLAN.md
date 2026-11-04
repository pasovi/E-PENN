# Paper 1 revision: independent assessment, run plan, and item 1 (R1)

*22 Oct 2026. Paper: "What Does a Game-Trained Embedding Learn?" (PDF of 8 Oct, 17 pp.).
Evidence comes from this repo. New code, results and figures are in `rev2/`; no original file
was modified.*

---

## 0. Bottom line

1. **I agree with the evaluation on almost every point I could check**, and I verified most of
   its checkable claims numerically (§1). The core idea is sound. §8 is a second paper. §9 should
   be replaced by verification against the closed-form V\*. The "winding class" is a heading bin.
   The statistics are underpowered and the run census does not add up.
2. **I found five further problems the evaluation does not mention** (§2):
   - the headline statistic PR(Z) is gauge-dependent;
   - adversarial training is chaotic, so seeds are not identifiers and the published baseline does
     not reproduce off its machine;
   - the arms used incompatible training protocols;
   - the baseline IC set is unrepresentative;
   - the "mediocre equilibrium" premise is mis-measured.
3. **Item 1 (R1: 120 runs under one deterministic protocol, with corrected metrics) is done.**
   It changes the paper's story, and for the better (§4):
   - **Before capture, game-trained strategies are exactly the constant-heading saddle of one
     image chart, which both players choose together.** Pre-capture constancy η = 0.9997;
     alignment with that chart's saddle direction is cos = 0.99999; the two players pick the same
     chart on 99.7% of heads; T_c/V\* = 1.010. All of the ensemble's excess over rank 2 is motion
     *after* capture, driven by the shaping term.
   - **The game's discrete content is the chart choice, and it is a real, non-circular result.**
     On capturable heads, 62 of 63 capture failures are plays in the wrong image chart. Chart
     errors fall from 48% at the cut locus to 1% far from it.
   - **The equilibrium is chart-local, not Nash.** A best-responding pursuer intercepts every
     committed evader, all 1,600 heads, by t ≤ 0.86, through a different chart 80–92% of the time.
   - **The init seed explains none of the run-to-run variance.** The IC set is the systematic
     factor, and the rest is indistinguishable from the spread produced by 1e-10 perturbations.
     The paper's "init dominates IC, 4.7:1" is reversed.
   - **The published baseline run is an outlier:** 98th percentile of PR(H_E), 5th percentile of
     capturable heads.
   - **The P/E spectral asymmetry is robust under the baseline protocol** (median ratio 1.86,
     ≥ 1.5 in 81% of runs), **but it lives in post-capture motion.**
4. **The plan (§3) is updated by R1.** The most useful next run is now a capture-truncated loss
   (R2). It tests the post-capture mechanism behind the rank excess, the asymmetry, and possibly
   part of the latent overhead. The game-free interventions on the latent come next.
5. Runs are cheap (about 3 min each with 8 in parallel; 100 runs ≈ 1 h). Compute is not the
   bottleneck; the rewrite is.

---

## 1. The evaluation, point by point

| Evaluation's claim | My verdict | Evidence |
|---|---|---|
| Core idea (ensemble vs latent, rank chain, named controls) is correct and valuable | **Agree, with caveats** | The rank chain bounds *rank*, not PR. PR is not monotone under linear maps, so PR(H) ≥ PR(Z) is an empirical fact, not a consequence. The "prediction confirmed" uses the wrong statistic (below). |
| §9 critic is stale; verify against closed-form V\* | **Agree** (the T² part) | V\* is closed form on T². The 31% / 23% figures measure grid generalisation, not play (§2.5). The claims about paper 2's drafts cannot be checked here (paper 2 is not in the repo). |
| §8 is a second paper; keep only the oracle arm | **Agree** | Its premise ("mediocre equilibrium") is also mis-measured (§2.5). |
| "Winding class" is a heading bin, not a homotopy class | **Agree, verified at scale** | w = round(2ū) in 99.1% of 1,600 heads (R1). Its clustering ratio (3.4×) is circular. Labels fixed before training give 1.5–2×; a permuted label gives 1.0× (§4.6). |
| Use the nearest-image chart m\* instead | **Agree, with a refinement** | m\* alone is still a heading-quadrant bin (pursuer at the origin). The non-circular observable is the chart each head *plays* versus m\*. R1 shows this is the game's real discrete content (§4.5). |
| 4.7:1 variance ratio not significant | **Agree; the reality is worse** | F(6,4) gives p = 0.078. At n = 100 the init seed explains 0% of the variance and the IC set 25% (§4.3). |
| Asymmetry decidable at 10 seeds | **Agree** | Under the baseline protocol it is robust (R1, n = 100). The open question is now its mechanism (§4.7, R2/R5). The repo holds 8 C2 runs, of which 4 are reported. |
| N_h sweep | **Agree** | R7 below. The existing `v3_nh*` checkpoints reach only 1,500 iterations. |
| Run census inconsistent | **Agree** | Also: C3 was trained to 15k, not the 14k stated. |
| PR must be the uncentred Gram; "ideal 2" → "upper bound 2" | **Agree, verified** | Saddle Gram PR = 1.92 [1.90, 1.96] over IC sets. The evaluation understates the problem: the headline statistic is not even gauge-invariant (§2.1). |
| 59.7% capturable share wrong | **Agree** | π·0.88²/4 = 60.8% (60.3% excluding the capture disk). |
| "At least 1.69" vs "ratio ≥ 1.69" | **Agree, and the claim itself fails at scale** | The ratio minimum over 120 runs is 1.49 (E) / 1.55 (P). |
| Actor is not physics-informed | **Agree** | The loss contains no PDE residual. |
| Citation fixes (Isaacs; Melikyan–Ovakimyan; Comm. Phys. 2025) | **Plausible, not checked** | Verify before changing. |
| Prose, abstract length, ~50 references | **Agree** | `REVIEW_claude.md` §7 has a vetted list of ~40 references. |
| Aggressive variant keeps the 8-seed baseline | **Disagree** | Those seeds mix protocols (§2.3). Reruns cost about an hour, so the aggressive timeline survives with reruns. |

---

## 2. Problems the evaluation does not mention

### 2.1 The headline PR(Z) is not gauge-invariant
The policy û^j = z^j/|z^j| is unchanged by z^j(t) → c_j(t) z^j(t), c_j > 0. Lemma 2 covers
constant c_j only. The time-centred PR of z, the paper's PR(Z), changes under this rescaling.
On the published baseline a smooth c(t) moves PR(Z_P) from 1.915 to 1.121, while the policy,
the temporal PR of u and the Gram PR do not move (`rev2/test_metrics.py`). Across R1, the
pursuer's PR(Z) (median 1.75) is systematically *below* its gauge-invariant counterpart (3.35).
**Fix:** use gauge-invariant statistics only:
- Gram PR for Proposition 1;
- the constancy η for "near-constant";
- the temporal PR of u for the time-varying part.

### 2.2 Adversarial training is chaotic, so seeds are not identifiers
Runs are bit-identical on one machine. A 1e-12 change to the initial weights gives
max |ΔB| = 0.81 after training. The published seed-0 baseline gives PR(H_E) = 6.00 / PR(Z_E) = 1.81
on this machine, against 6.75 / 2.99 published. R1 quantifies the consequence (§4.3): single-run
numbers are draws, and "variance set by the initialisation" confuses initialisation with chaos.

### 2.3 Protocol census of the existing arms

| arm | iterations | Adam reset | warm-up |
|---|---|---|---|
| baseline `v2_actor` | 14k | every 3,500 | 1,500 |
| C1 init seeds `c1_s*` | **10k** | every 33 s of wall time (CPU-dependent) | 1,500 |
| C1 IC seeds `c1_ic*` | **10k** | every 2,500 | 1,500 |
| C2 original / seeds | 28k | 33 s / 2,500 | none |
| C3 `c3_k6/k24` | **15k** | every 2,500 | 1,500 |
| C4, C5, Δv sweep | 14k | every 2,500 | 1,500 |
| C7 game phase | 14k | 33 s | none |
| protocol study | 5k | none | 800 |

The paper's "eight init seeds" combine one 14k/3,500 run with seven 10k/33-s runs, and the IC
arm is a third protocol. R1 shows the reset cadence itself barely matters (§4.8), but the
iteration counts and the non-reproducible chunking must go.

### 2.4 The baseline IC set is unrepresentative
Capturable heads per IC seed 0–19: 4, 7, 9, 9, 8, 12, 7, 9, 7, 10, 11, 8, 12, 7, 10, 8, 11, 5, 9, 6
(mean 8.45). Seed 0 is the least capturable of the twenty, and every "capturable heads" statistic in
the paper rests on 4 curves.

### 2.5 The "mediocre equilibrium" is mis-measured
- The 23% (and the critic's 31%) comes from `capture_time_map`. It plays the nearest trained
  head's open-loop controls from each point of a 24×24 IC grid. That tests generalisation to
  untrained ICs, not equilibrium play.
- On its own ICs the self-play pursuer captures 92% of capturable heads (100% at the baseline)
  at T_c/V\* = 1.010. A best responder captures 100% of all heads by t ≤ 0.86 (§4.4).
- "Mediocre" should become "chart-local equilibrium". Exploitability is exact and closed form on
  T². Every published actor checkpoint shows the same 16/16 best-response capture
  (`rev2/results/legacy_corrected.json`).

### 2.6 Smaller technical points
- Centring H removes its time mean, which is the direction that produces the constant part of the
  headings. Only time-indexed spectra (PR(H), temporal PR of u) should be compared through the
  rank chain. The Gram PR is head-indexed and is a different object.
- Proposition 1 is trivially true for constant curves; the empirical content is the constancy.
  Measure it directly (η, and per capture window: §4.2).

---

## 3. Run plan, ordered by urgency and usefulness (revised after R1)

Every arm uses the deterministic protocol of `rev2/core.py` (with chunking as an explicit config
field) and the metrics of `rev2/metrics.py`. Each arm has at least 10 runs, is reported as
distributions, and has a decision rule fixed before it runs. Costs are wall-clock on this laptop
(M3, 8 workers): about 30 s per baseline-size run, amortised.

### R1. Baseline ensemble with corrected metrics — **DONE** (§4)
- **Design:** 10 init seeds × 10 IC sets, plus a chaos arm (10 perturbations of size 1e-10)
  and a no-reset arm (10). 120 runs, 59 min.
- **Status:** feeds the new main figure, the census table, the variance decomposition replacing
  C1, the closed-form verification replacing §9, the chart analysis replacing §5, and the
  baseline null for every later arm.

### R2. Capture-truncated loss — *new, highest priority*
- **Design:** replace the shaping term λ_d·⟨d⟩_t by λ_d·⟨S(t) d(t)⟩_t, with the exact gradient
  through S and a finite-difference check. Nothing after capture then affects either loss. Run
  on the same 10 × 10 init/IC grid as R1 for a paired comparison: 100 runs, about 1 h, plus about
  half a day to implement.
- **Why:** R1 shows that all time variation of the strategies is post-capture (η_post ≈ 0.15–0.18
  against η_pre = 0.9997), and that the P/E asymmetry lives in captured heads. A plausible
  mechanism is kinematic: a fixed-speed pursuer that is 20% faster must weave to stay within ε of
  a slower evader. R2 tests three headline claims at once:
  - whether the rank excess is shaping-driven post-capture motion;
  - whether the asymmetry is;
  - whether part of the latent's temporal modes exists to support that motion.
- **Rules (paired Wilcoxon against R1):**
  - **Asymmetry:** attributed to post-capture shaping if the median band ratio falls to ≤ 1.2,
    or if post-capture power becomes untrained noise that is equal for P and E.
  - **Overhead:** if PR(H) is unchanged (median shift inside R1's IQR), it is not post-capture
    machinery. A significant drop means part of it was.
  - **Strategy:** pre-capture constancy should stay ≥ 0.999. Otherwise the post-capture motion
    was also shaping the play itself.

### R3. Game-free interventions on the latent overhead (decisive test of §4)
- **Design:**
  - (a) Regress the same architecture onto the analytic constant headings, with no game, the same
    optimiser and the same number of updates per player, on the 10 × 10 grid (100 runs).
  - (b) Distil each R1 actor's own pre-capture headings from the same init (100 runs).
  - (c) Body-init gain g ∈ {0.25, 0.5, 2, 4} in the game, 10 runs each.
  - About 1.5 h in total.
- **Why:** "featurisation residue" rests on correlational controls. One of them, the variance
  split, has been overturned by R1. An intervention that removes the game settles it.
- **Rule:** if the game-free PR(H) median lies inside R1's IQR, attribute the overhead to
  architecture plus optimiser. If it is lower with p < 0.01 and outside the IQR, the game adds
  latent modes and §4 is rewritten. For (c), featurisation predicts PR(H) moves with g while the
  ensemble statistics stay flat.

### R4. Warm-start collapse (C7) at 10 seeds
- **Design:** plant curved families at Gram PR ≈ 4 and ≈ 9, 10 seeds each, then game-train.
  Add the PR-9 level at λ_d = 0 (10 seeds). Track Gram PR, pre-capture η and the played chart
  every 250 iterations. 30 runs, about 30 min.
- **Why:** "selected, not default" is the claim that keeps the rank result from being trivial.
  It is currently n = 1. On the published C7 checkpoint the corrected metric confirms the
  collapse: Gram PR 10.7 → 2.3, η 0.12 → 0.89.
- **Rule:** collapse if the final Gram PR lies within R1's IQR in ≥ 8/10 seeds per level.

### R5. Asymmetry: symmetrised protocol and single-factor ablations
- **Design:** the fully symmetrised protocol, 20 runs to 28k iterations with snapshots, plus
  four single-factor ablations from the baseline (symmetric loss, no warm-up, 1:1 steps, equal
  learning rates), 10 runs each. About 1.3 h.
- **Why:** the paper's attribution question (inf–sup vs protocol). If R2 removes the asymmetry,
  this reduces to confirming that under the symmetric protocol, and the ablations can be dropped.
- **Rule:** the paper's own rule on the run-median band ratio, with a bootstrap CI.
  - Median ≥ 1.5 with the CI above 1.2: attributable to inf–sup.
  - Median ≤ 1.2 with the CI below 1.5: protocol artefact.
  - Anything else: report as unresolved, with the CI.

### R6. Featurisation sweeps
- **Design:** K ∈ {3, 6, 24, 48} and body width n_b ∈ {8, 32, 64}, 10 runs each. 70 runs,
  about 1 h.
- **Why:** "the break tracks K" currently has no figure, and n = 1 per K.
- **Prediction:** a break at k ≈ K; PR(H) rising with K and n_b; ensemble statistics flat.

### R7. Heads sweep
- **Design:** N_h ∈ {8, 32, 64, 128}, 10 runs each. About 1.7 h (cost scales with N_h).
- **Why:**
  - It tests whether the overhead grows with the number of readers.
  - It fills in the chart-error-vs-cut-locus curve with 5–8× more heads.

### R8. Δv sweep at 10 runs per value, both players
- **Design:** v_P ∈ {1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50}, with ICs varying. 80 runs,
  about 1 h.
- **Why:** it currently has 3 seeds, the evader only, the legacy statistic and a fixed IC set.
  Add capture vs V\*, chart errors and exploitability across ρ = 1. The weaving mechanism (R2)
  predicts that the asymmetry scales with the speed excess.

### R9. Shaping sweep
- **Design:** λ_d ∈ {0, 1, 2, 10}, 10 runs each, with gradient shares measured at the end.
  40 runs, about 30 min.
- **Why:** the shaping disclosure at proper n. The published λ_d = 0 run is the most constant
  (η = 0.97 / 0.99). Does the chart choice survive without shaping?

### R10. Oracle arm and the full Nash gap
- **Design:**
  - (a) An oracle pursuer against the fixed saddle evader, 10 runs.
  - (b) An evader best response to each R1 pursuer, via reachability on a torus grid (analysis
    only).
- **Why:** "rank two under stronger play" currently rests on n = 3. (b) completes the
  exploitability picture whose pursuer side R1 measured.

### R11. (Optional, appendix only) Protocol study at 10 seeds
Only if §8 survives as an appendix, at a budget matched to R1.

**Total for R2–R10:** about 10 h of laptop time.

**Not runs, but needed:**
- a schematic of the game, the architecture and the image charts;
- a human rewrite built from the new figures and tables;
- moving §8 to an appendix;
- replacing §5 ("topology as clusters") by the chart-selection result.

---

## 4. Item 1 results: R1 baseline ensemble

**What was run.** `rev2/run_r1.py`, 120 runs at the published configuration (14,000 self-play
iterations after a 1,500-step pursuer warm-up, 3:1 steps, Adam 1e-2/1e-3 reset every 3,500
iterations, λ_d = 5, λ_nc = 2):

| arm | runs | varied |
|---|---|---|
| grid | 100 | init seed 0–9 × IC seed 0–9; cell (0, 0) is the published configuration |
| chaos | 10 | init 0, IC 0, plus 1e-10 · N(0, 1) on every initial parameter |
| no-reset | 10 | init 0–9, IC 0, with one Adam for the whole run |

- Wall time was 59 min.
- `rev2/test_metrics.py` gates the metrics: it reproduces all 13 published legacy numbers from
  `v2_actor.npz`.
- `rev2/analyze_r1.py` regenerates every number below:
  - `rev2/results/r1_summary.json`;
  - `rev2/results/r1_census.md` (full census table, all three arms);
  - `rev2/figs/r1_*.{png,pdf}`.

### 4.1 Census (grid, n = 100): median [IQR]

| statistic | evader | pursuer | published run (percentile) |
|---|---|---|---|
| Gram PR (Proposition 1's object) | 2.47 [2.32, 2.70] | 2.50 [2.30, 2.69] | 2.54 / 2.48 (59 / 46) |
| … analytic saddle, same IC sets | 1.92 [1.90, 1.96] | | 1.96 |
| constancy η (full horizon) | 0.852 [0.813, 0.886] | 0.848 [0.816, 0.891] | 0.85 / 0.87 |
| temporal PR of u (gauge-invariant) | 2.80 [2.39, 3.24] | 3.35 [2.85, 3.78] | 2.85 / 3.14 |
| temporal PR of z (legacy "PR(Z)") | 2.43 [1.95, 2.74] | 1.75 [1.57, 2.04] | 2.99 / 1.91 (91 / 65) |
| PR(H), centred (legacy) | 5.73 [5.29, 6.13] | 6.51 [5.98, 7.28] | 6.75 / 7.08 (**98** / 67) |
| PR(H), uncentred | 3.22 [2.76, 3.57] | 3.97 [3.45, 4.73] | 4.21 / 4.02 |
| latent power at k ≤ K | 0.984 | 0.913 | |
| PR(H) / temporal PR of u, within run | 2.04 [1.77, 2.30] | 1.96 [1.71, 2.29] | |
| capturable heads (of 16) | 8.5 [7, 9] | | 4 (**5.5th**) |

- **The legacy statistic misleads mostly for the pursuer.** Its "≈ 2" is 3.35 once the
  gauge is removed.
- **The published run sits in the tails** of PR(H_E) and of the capturable count. It is not a
  typical draw.
- **Within-run separation:** latent temporal modes exceed policy temporal modes in 100% (E) and
  99.2% (P) of the 120 runs, by a factor of about 2. The paper's "≥ 1.69 in every checkpoint"
  fails at scale: the legacy-ratio minimum is 1.49 (E) / 1.55 (P), and the gauge-invariant
  minimum is 1.30 / 0.88. The "two populations do not overlap" claim holds only for the legacy
  pair.

Figure: `r1_dimensionalities` (every run, every statistic, published run marked).

### 4.2 Where the time variation lives: post-capture motion
Each captured head is split at its capture time (`r1_capture_windows`):

| window | η, evader | η, pursuer |
|---|---|---|
| uncaptured heads, full horizon (n = 842) | 0.99999 | 1.00000 |
| captured heads, **before** capture (n = 758) | 0.9997 | 0.9998 |
| captured heads, **after** capture (n = 718) | 0.18 | 0.15 |

- **During play, the strategies are constant headings to four decimals.** All of the ensemble's
  excess over the saddle comes from the post-capture window: Gram PR 2.5 vs 1.9, η = 0.85, the
  temporal PR of u. In that window the survival term no longer depends on the headings, but the
  shaping term λ_d⟨d⟩ still does, so post-capture motion is a close-range pursuit at 1.2 vs 1.0.
- Consistently, the Gram PR on capturable heads (2.68) exceeds that on uncapturable heads
  (1.85 ≈ saddle).
- **The paper's "soft-game corrections" are post-game motion.** The precise statement for the
  paper: before capture the ensemble *is* Proposition 1's object.

### 4.3 Where run-to-run variation comes from
Two-way random-effects ANOVA on the 10 × 10 grid, plus the chaos arm (`r1_variance`):

| statistic | init seed (p) | IC set (p) | share init / IC / residual | chaos var / residual var |
|---|---|---|---|---|
| PR(H) evader | 0.45 | 1e-4 | 0% / 25% / 75% | 1.15 (p = 0.67) |
| PR(H) pursuer | 0.52 | 0.05 | 0% / 9% / 91% | 0.88 (p = 0.89) |
| Gram PR evader | 0.42 | 8e-10 | 0% / 46% / 54% | 0.18 |
| Gram PR pursuer | 0.49 | 2e-13 | 0% / 57% / 43% | 0.23 |
| constancy η, pursuer | 0.26 | 5e-17 | 1% / 65% / 35% | 0.22 |
| band ratio P/E | 0.57 | 0.53 | 0% / 0% / 100% | 0.18 |

- **The init seed has no detectable systematic effect on any statistic**: p = 0.10–0.82
  across all 14 tested.
- **The IC set is the systematic factor.** IC-set means of the Gram PR track the number of
  capturable heads (Spearman 0.85 / 0.81), and constancy tracks it inversely (−0.91).
- **For the latent PR, the residual is indistinguishable from the chaos floor.** A 1e-10
  perturbation of one init produces as much spread as all the remaining variation.
- For ensemble statistics the chaos arm is quieter than the residual. It sits at the
  4-capturable IC set, where most heads are uncapturable and constant.
- **The published C1 conclusion is reversed.** Resampling the published design from the grid
  (8 inits at one IC vs 5 ICs at one init) gives a median ratio of 0.73, a 90% interval of
  [0.10, 3.9], and P(ratio ≥ 4.7) = 3.5%.

### 4.4 Equilibrium quality, verified in closed form (replaces §9)
`r1_game`, 1,600 heads:
- **The trained pursuer:**
  - captures 92.3% of capturable heads (95.8% median per run) and 0.1% of uncapturable ones;
  - captures at **T_c/V\* = 1.010 [1.004, 1.015]**, with dt = 0.0625;
  - always captures in the nearest-image chart.
- **A best-responding pursuer** (exact straight-line interception of the committed evader path)
  captures **100% of heads, including all 780 uncapturable ones, by t ≤ 0.86**. On capturable
  heads, T_BR/V\* = 0.24 [0.18, 0.30], and it intercepts through a chart other than m\* in 92% of
  them.
- **Reading:** self-play converges to the open-loop realisation of the *feedback* saddle within a
  chart. This is a chart-local (differential) equilibrium of an open-loop game that has no value.
  It is not a Nash equilibrium.

### 4.5 The game's discrete content is the chart choice (replaces §5's winding analysis)
`r1_charts`. The played chart of a head is the image m that maximises cos(⟨u⟩_pre-capture,
Δ_m), with collinear ties going to the nearer image.
- **Every head plays a pure saddle in some chart:** median cos 0.99999, minimum 0.978.
- **Both players choose the same chart on 99.7% of heads.**
- **The pursuer plays the nearest chart m\* on 82% of all heads and 92% of capturable heads.**
- **Capture is decided by the chart:** capturable heads in m\* are captured 99.87% of the time
  (755/756); heads in another chart 3% (2/64). **62 of the 63 capture failures are chart errors.**
- **Chart errors sit at the cut locus.** The error rate by octile of the chart gap
  |Δ₍₂₎| − |Δ₍₁₎| runs from 48% (gap < 0.08, a coin flip between two nearly equidistant images)
  down to 1% (gap > 0.55). The figure shows deciles.

This is a non-circular, topological result. It ties directly to the chart construction of paper 2.

### 4.6 The paper's winding clusters are circular
Mean / median across-class over within-class L² distance, pooled over 100 runs:

| label | ratio | within-class pairs |
|---|---|---|
| winding, the paper's label | 3.45 / 3.94 | 631 |
| IC heading bin round(2Δ̂₀), fixed before training | 1.55 / 2.01 | 730 |
| nearest-image chart m\* | 1.46 / 1.62 | 3,030 |
| permuted winding (null) | 1.01 / 1.01 | 631 |

The winding label equals round(2ū) on 99.1% of heads, so it bins the very quantity whose
distances are measured. The paper's 6× (mean) and 20× (median) came from 5 pairs.

### 4.7 The asymmetry statistic under the baseline protocol
- **The asymmetry is robust at population level.** The band fraction ratio P/E for
  k ∈ [11, 50] is 1.86 [1.59, 2.16]. It is > 1 in 99% of runs, ≥ 1.5 in 81%, and ≤ 1.2 in 4%.
- **It is unstable run by run.** The Spearman correlation between a run's value at 10.5k and at
  14k is 0.04.
- **It is carried by captured heads:** 1.86 on captured heads, against 1.50 [0.88, 2.71] on
  uncaptured heads, whose time-varying power is tiny. That points to post-capture motion
  (R2, R5).
- The raw singular-value ratio P/E is 3.08 [2.69, 3.48], the gauge artefact of Lemma 2, as
  published.

### 4.8 Protocol robustness and stationarity
- **Adam resets:** one Adam vs a reset every 3,500 (paired, n = 10) shows no difference in any
  ensemble or game statistic (p ≥ 0.27). Latent PR shifts by ±0.55 in opposite directions for
  E and P (p = 0.014 / 0.049), which does not survive correction for 16 tests.
- **Population statistics converge by 3.5k iterations.** Median Gram PR is 2.39 → 2.47 and
  median PR(H_E) is 5.84 → 5.73 between 3.5k and 14k.
- **Single runs keep wandering.** The run-level Spearman between 10.5k and 14k is 0.4–0.7 for PR
  statistics. This is consistent with a non-stationary self-play equilibrium, and is another
  reason to report distributions.

### 4.9 What this means for the paper
| Section | Change |
|---|---|
| Abstract and §3 | Lead with "before capture, game-trained strategies are the constant-heading saddle of a jointly chosen image chart" (η = 0.9997, cos = 0.99999). The rank-2 statement follows and becomes non-trivial through C7/R4. |
| §5 | Replace by the chart-selection result (§4.5): misses are chart errors, concentrated at the cut locus. |
| §9 | Replace by closed-form verification (§4.4): T_c/V\* = 1.01; chart-local, exploitable equilibrium. |
| §4 | The latent overhead is real (latent modes ≈ 2× policy modes in 99–100% of runs), but its variance is IC plus chaos, not init. Whether it is featurisation residue is R3's question; R2 checks whether part of it serves post-capture motion. |
| §7 | The asymmetry is robust but post-capture; its mechanism is the open question (R2/R5). |
| Throughout | Drop PR(Z); report distributions over the 100-run grid; state reproducibility as "distributions reproduce; single runs reproduce only on identical hardware and software". |

### 4.10 Files
| path | contents |
|---|---|
| `rev2/core.py` | deterministic training harness; `RunConfig` holds every protocol choice |
| `rev2/metrics.py` | every diagnostic, each defined by a formula in its docstring |
| `rev2/test_metrics.py` | gate reproducing the 13 published legacy numbers, plus gauge and Proposition 1 checks |
| `rev2/run_r1.py`, `rev2/analyze_r1.py` | R1 runner and analysis |
| `rev2/legacy_checkpoints.py` | every published checkpoint re-scored with the corrected metrics (`rev2/results/legacy_corrected.json`) |
| `rev2/results/r1/*.json`, `rev2/ckpt/r1/*.npz` | per-run metrics (light metrics every 1k iterations; full metrics at 3.5k/7k/10.5k/14k) and parameters at those four checkpoints |
| `rev2/figs/r1_*.{png,pdf}` | dimensionalities, capture windows, charts, game, variance, spectra, dynamics |
