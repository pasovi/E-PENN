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