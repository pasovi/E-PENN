# Orientation notes: code map, model, vocabulary, baseline

Written 28 Oct 2026 against the 8 Oct PDF ("GameTheoryEmbeddings (10).pdf") and this bundle.
Figures referenced below are in `viz/baseline/` (published actor) and `viz/pilot_*/` (minimal game).

## 1. What the network is (and is not)

**Inputs: time only.** `phi(t) = [sin(2πk t/t_max), cos(2πk t/t_max)]_{k=1..K}, t/t_max`, K = 12,
so 25 input features (`pursuit_evasion_numpy.py: fourier_features`). The network never sees a
position, a displacement, or the opponent. The evader initial condition enters only through
*which head* is used: head j is trained on IC j. That is what "open-loop" means here.

**Body:** one tanh layer, `H(t) = tanh(B phi(t) + b)`, 16 units. This 16-dim curve H(t) is the
"latent curve" / "embedding" the paper analyses (`forward_dirs`, cache key `H`).

**Heads:** 16 linear maps `W^j ∈ R^{2×16}`, one per IC: `z^j(t) = W^j H(t)`, then normalised,
`û^j(t) = z^j/|z^j|`, a unit heading in R². Fixed speed, so the heading is the whole control.
Each player (E, P) has its own body + 16 heads; the two players share nothing.

**Outputs:** 16 unit-heading curves per player. Positions are *not* outputs: they come from
rolling the kinematics forward inside the loss.

**Rollout (the differential equation):** `ẋ_E = v_E û_E(t)`, `ẋ_P = v_P û_P(t)` on the flat
torus `[0,2)²`, integrated with an explicit midpoint/trapezoid step, 64 steps over t_max = 4,
with wrap (`rollout`). The backward pass is hand-coded (`backward_rollout`, `backward_dirs`,
`backward_loss`) and checked against finite differences. So yes, an ODE is being solved, but
only as a forward simulator inside the loss (direct single shooting with a neural control
parametrisation). No PDE residual appears in the actor's loss. **The actor is not a PINN**; the
word is inherited from the Burgers paper's architecture. The only PINN in the bundle is the
critic `V(Δ)` (`pursuit_evasion_v2.py: hji_residual_loss`, and `honest_critic_geom.py`), trained
on the HJI residual `(v_E − v_P)|∇V| + 1 = 0`; on T² its solution is closed form,
`V* = (|Δ|_T − ε)/(v_P − v_E)`, which is why it only serves as a check.

**Loss** (`forward_loss`): per head, torus distance `d(t)`, soft capture probability
`p = σ((ε² − d²)/τ)` with τ = 5e-3, survival `S(t) = Π(1 − p)`, soft capture time
`T̂ = ∫ S dt`. Then
`L_P = T̂ + λ_nc S(t_max) t_max + λ_d ⟨d⟩`, `L_E = −T̂ − λ_d ⟨d⟩`, λ_nc = 2, λ_d = 5.
Because τ is tiny, the `T̂` term has essentially zero gradient until a head is already within
~0.1 of capture. The `⟨d⟩` term is what actually trains (the paper measures: ~100% of the
evader's gradient, 6% of the pursuer's).

**Training protocol** (`run_v2.py: make_actor_cfg`, canonical form in `rev2/core.py: RunConfig`):
1,500 pursuer-only warm-up steps, then 14,000 iterations of 3 pursuer Adam steps (lr 1e-2) per
1 evader step (lr 1e-3), Adam state reset every 3,500 iterations. The published baseline is
`v2_actor.npz` (seed 0 for both init and ICs).

## 2. Where each piece of the paper lives

| Paper | Code | Data |
|---|---|---|
| §2 game, architecture, loss | `pursuit_evasion_numpy.py` (everything), `run_v2.py` phase 1 (baseline training) | `v2_actor.npz` |
| §3 PR of latent vs strategy ensemble; "rank two under stronger play" | `embedding_analysis.py` (PCA/PR of H), `oracle_pr.py` (oracle pursuer, head PR), `pr_capturable_scrutiny.py`, `revision/post_hoc_checks.py` | `embedding_summary.json`, `pedro_oracle_pr_result.json` |
| Table 1 C7 warm-start collapse | `run_c7.py` | `c7_fit.npz`, `c7_actor.npz` |
| Table 1 C5 shaping sweep, C4 sector | `run_c4c5.py`, `revision/analyze_e1e7.py` | `c5_lam*_actor.npz`, `c4_actor.npz`, `revision/c4c5_analysis.json` |
| Table 1 C3 bandwidth sweep, Fig. 1 | `run_c3.py`; no committed plotter for Fig. 1 (see REVIEW_claude.md) | `c3_k6_actor.npz`, `c3_k24_actor.npz`, `revision/c3_break_analysis.json` |
| Table 1 C1 init seeds / IC resamples | `run_c1.py`, `run_c1_ic.py` | `c1_s*.npz`, `c1_ic*.npz`, `revision/c1_seeds.json` |
| Table 1 C2 symmetrised (exact zero-sum) | `run_c2.py`, `run_c2_seeds.py` | `c2_*.npz`, `revision/c2_seed_replicates.json` |
| §5 winding / sectors, Figs. 2–3 | `run_e17.py` (E1 sector, E7 anneal), `run_e1b_reflected.py`, `winding_analysis.py`, `make_fig_sectors.py`, `make_fig_winding_reflection.py` | `e1*_actor.npz`, `e7_*.npz`, `revision/e1e7_analysis.json` |
| §6 Δv sweep, Fig. 4 | `run_dv_sweep.py`, `analyze_dv_sweep.py` | `dv_sweep_results_full.jsonl` (no checkpoints) |
| §7 gauge lemma, Fig. 5 | `embedding_analysis.py: head_weight_singular_values`; `rev2/metrics.py: head_stats` | `v2_actor.npz` |
| §8 protocols, honest corrections, Shapley, Figs. 6–8 | `run_game_protocols.py`, `run_honest_protocols.py`, `shap_attribution.py`, `analyze_*.py` | `*_results*.jsonl`, `shap_seed*.json` |
| §9 critic | `pursuit_evasion_v2.py` (critic), `pursuit_evasion_v4.py`, `run_v4.py`, `run_v5.py`, `honest_critic_geom.py`, `run_geom.py`, `revision/post_hoc_checks2.py` | `v2_V.npz`, `v4_honestV.npz`, `v5_honestV.npz`, `geom_*.npz` |
| Revision (not in the PDF) | `rev2/core.py` (deterministic protocol), `rev2/metrics.py` (gauge-invariant metrics), `rev2/run_r1.py`, `rev2/analyze_r1.py` | `rev2/results/r1/*.json` (120 runs), `rev2/results/r1_census.md` |
| Visualisation (new) | `viz/visualize_actor.py`, `viz/minimal_game_pilot.py` | `viz/baseline/`, `viz/pilot_*/` |

Read `REVIEW_claude.md` and `New_Revision_plan.md` before trusting any number in the PDF: both
document corrections (the headline PR(Z) is not gauge-invariant, the seeds mix protocols, the
published baseline is an outlier, "winding class" is a heading bin, the actor is not a PINN).

## 3. Vocabulary

- **Actor**: the self-play object, i.e. the pair (evader body+heads, pursuer body+heads). Not
  the RL actor; nothing feeds it from the critic.
- **Critic**: the separately trained value network `V(Δ)` (HJI residual). Only used post hoc.
- **Oracle / oracle-trained pursuer**: a pursuer trained against an evader *fixed* at the
  analytic optimum (constant heading along the min-image direction). "Oracle" because it uses
  the analytic solution, which the "honesty contract" forbids in the paper's own runs.
- **Honesty contract**: no analytic value or control enters any loss. The disclosed exception:
  on T², `⟨d⟩` is an affine image of `V*`, so distance shaping *is* value shaping.
- **Open-loop**: control is a function of time only, `u(t)`, committed at t = 0. **Closed-loop /
  feedback**: `u(x)`. Isaacs's saddle is a feedback saddle; here the heads realise its open-loop
  trace for one IC each. The paper notes the open-loop game has no value (lower value 1.11 <
  upper value 1.66), so self-play converges, if at all, to a protocol-selected point.
- **Heading**: unit direction of motion `û`. **Control curve**: one head's `û^j(·)`.
  **Strategy ensemble `S_α`**: the 16 control curves of player α. **Latent curve `H_α(t)`**:
  the body output.
- **Saddle / constant-heading saddle**: both players move along `Δ̂_0 = Δ_0/|Δ_0|`, the
  min-image displacement direction; pursuer chases, evader flees, straight lines.
- **Winding `w ∈ Z²`**: the evader's lifted net displacement over `[0, t_max]` divided by L,
  rounded. Sold as a homotopy class; since `displacement = v_E t_max ū = 4ū` and L = 2, it is
  `round(2ū)`, i.e. a bin of the mean heading (rev2 §4.6). **Sector**: ICs restricted to an
  angular wedge. **Winding class / cluster**: heads with equal w.
- **Chart / image chart `m ∈ Z²`** (rev2): which periodic image of the pursuer's start the
  pursuer actually reaches; `m*` is the nearest image. "Chart errors" = playing the wrong image.
- **Cut locus / antipodal ridge**: points at `|Δ_i| = L/2`, where the nearest image is ambiguous.
  12 of the 16 baseline ICs sit within 0.3 of it.
- **Reach radius `R = (v_P − v_E) t_max + ε = 0.88`**; **capturable head**: `|Δ_0| ≤ R`
  (4 of 16 at seed 0). **ρ = R/(L/2)**; ρ = 1 is where the reach disk touches the cut locus.
- **Participation ratio `PR = (Σλ)²/Σλ²`** of a PSD spectrum: an "effective number of modes".
  Three different PRs appear: `PR(H)` = PR of the time-covariance of H (16×16); `PR(Z)` = PR of
  the time-covariance of the 32-dim stacked head outputs (gauge-dependent, see rev2); Gram PR
  = PR of the 16×16 Gram matrix of the unit-heading curves across heads (the object
  Proposition 1 bounds by 2).
- **Gauge**: `W^j → c_j W^j`, c_j > 0, leaves `û^j` unchanged. **Gauge fixing**: unit Frobenius
  norm per head. **Effective rank** = PR of squared singular values of the stacked heads.
- **Featurisation residue**: extra latent dimensions attributed to Fourier features + tanh +
  init rather than to the game. **Spectral break at K**: latent power drops above input band.
- **Shaping**: the `λ_d ⟨d⟩` term. **Survival bonus / no-capture penalty**: `λ_nc S(t_max) t_max`
  (pursuer only in the baseline; mirrored in C2).
- **Protocol**: warm-up, step ratio (3:1), learning-rate ratio (10×), Adam resets. **Symmetrised
  protocol (C2)**: exact zero-sum loss, no warm-up, 1:1, equal lr.
- **GDA / OGDA / EG / ABR**: simultaneous gradient descent-ascent, optimistic GDA,
  extragradient, alternating best response. **Honest protocols**: those without the oracle.
- **Band / intermediate band `k ∈ [11, 50]`**: temporal Fourier index band of heading power
  used for the P/E asymmetry statistic.
- **Constancy η = |⟨û⟩_t|²** (rev2): 1 iff the heading is constant in time.

## 4. What the runs look like (viz/baseline, published actor)

- `fig_torus_trajectories.png`: 16 panels. On every head both players move in a straight line
  along the saddle direction until capture; on the 12 uncapturable heads they stay on that line
  for the whole horizon. The three captured heads then show the pursuer weaving around the
  evader: a 20%-faster player with fixed speed cannot stay within ε without turning.
- `fig_headings.png`: heading angle vs time. Pre-capture constancy η_pre ≥ 0.994 on every head
  (table in `head_table.md`); post-capture η drops to 0.01–0.3. All the "time variation" of the
  strategy ensemble is post-capture motion driven by the shaping term, exactly as rev2 R1 found.
- `fig_lifted.png`: the same paths on the universal cover; the image of the origin that each
  pursuer path ends at is the chart it plays. The 4 captured heads all play the nearest image.
- `fig_torus3d.png`: a few heads on the embedded donut.
- `fig_latent.png`: `H_E(t)`, `H_P(t)`, their PCA time profiles and spectra. The pursuer's
  latent is saturated (±1 with switches), a consequence of the 10× learning rate; this is what
  the raw head singular-value asymmetry (Fig. 5 left) reflects.

## 5. PR vs PCA, S vs H, S¹/T⁴

- PR is not an alternative to PCA. It is a scalar summary of the PCA eigenvalue spectrum,
  the same spectrum `embedding_analysis.py` plots. The paper also reports the 90%-variance count
  (4 of 32 for the ensemble, 3 of 20 for Burgers). Use both; argue about *which covariance*.
- The paper measures PR on both objects. The result is the *comparison*: PR(H) ≈ 5–7 vs
  Gram PR(S) ≈ 2. H is the object of interest, but on its own PR(H) cannot say whether its
  dimensions are game content. The ensemble is what the game constrains, and the heads read
  only a rank-2 slice of H. One subtlety: time-centring H removes its time-mean, which is the
  direction that produces the constant headings, so PR(H) counts only the time-varying part of
  the latent, i.e. exactly what a constant-heading strategy never uses. The uncentred PR(H)
  is ≈ 3–4 (`rev2/results/r1_census.md`).
- Constant headings in R^d span ≤ d, so Gram PR ≤ d on T^d: ≤ 1 on S¹ (heading is ±1), ≤ 3 on
  T³, ≤ 4 on T⁴. Repeating on another d would show that "2" is the ambient control dimension,
  not an arbitrary number. The bundle's actor is hard-coded to d = 2 (`init_params` W shape,
  `rollout` allocation, IC sampler); the critic already supports T^d
  (`pursuit_evasion_v2.py: demo_3d_torus`). It is a small generalisation. Note the warm-start
  control C7 and the pilot below already address the sharper question (is rank 2 selected
  by the game rather than built in).

## 6. The baseline, and the simplest game (viz/pilot_*)

The published baseline has: three loss terms, a one-sided survival bonus, a pursuer-only
warm-up, 3:1 steps, 10× learning rates, Adam resets, τ = 5e-3 (so the capture-time term is
inert), and an IC set with 12/16 heads at the cut locus and 4/16 capturable. Exact zero-sum
(run_c2.py) removes the protocol asymmetries but keeps all three terms.

Pilot (`viz/minimal_game_pilot.py`): single-term, exactly zero-sum, open-loop game
`J = ⟨d_T(x_E, x_P)⟩_t`, `L_P = J`, `L_E = −J`, no capture mechanics, no warm-up, 1:1
alternating Adam at equal lr 3e-3, same architecture and the same seed-0 ICs, 3,000 iterations
(4 s). Off the cut locus the open-loop saddle of this game is exact: both players at constant
heading along `Δ̂_0`.

| run | Gram PR E/P | η E/P | PR(H) E/P (centred) | PR(H) uncentred | heads at ±saddle |
|---|---|---|---|---|---|
| init (seed 0) | – | – | 8.7 / 8.7 | – | – |
| integrated d, t_max = 1.5 (no head reached) | 2.01 / 2.02 | 0.993 / 0.988 | 6.97 / 7.38 | 3.1 / 4.1 | 13/16 at |cos| = 1.000; the 3 within 0.1 of the cut locus play the other chart, E and P agreeing (cos(E,P) = 1 on all 16) |
| terminal d, t_max = 1.5 | 2.00 / 2.21 | 0.990 / 0.946 | 7.6 / 6.8 | 3.3 / 5.1 | 11/16 |
| integrated d, t_max = 4 (4 heads reached) | 4.3 / 3.4 | 0.62 / 0.70 | 6.6 / 6.7 | 6.1 / 5.0 | 8/16 |
| published baseline (for reference) | 2.37 / 2.33 | 0.85 / 0.85 | 6.75 / 7.08 | 3.2 / 4.0 | all 16 pre-capture |

Three conclusions, each from one line of the table:
1. The simplest zero-sum game reproduces the paper's central result with nothing to argue
   about: Gram PR = 2.0, η = 0.99, exact saddle alignment, and the only "discrete content" is
   the chart choice at the cut locus, which both players make together (rev2's finding).
2. The latent overhead is there too: PR(H) ≈ 7 with no capture, no shaping term, no protocol
   asymmetry, and 8.7 at initialisation. The game removes little of it. That is the
   "featurisation residue" claim, obtained by intervention rather than correlation (what rev2
   plans as R3).
3. Let a head be reached inside the horizon under a fixed-speed, non-terminating rollout and
   constancy collapses (η 0.6–0.7, Gram PR 3.4–4.3): the pursuer must weave. The paper's
   "soft-game corrections" above rank 2 are this kinematic artefact, not strategy.

Recommended baseline ladder (each level adds one thing):
- **L0**: the pilot as is, 1:1, equal lr, integrated distance; horizon or ICs chosen so no
  head is reached (`|Δ_0| > v_P t_max`) and ICs kept ≥ 0.2 from the cut locus. Use ≥ 32 heads
  (the PR over 16 curves has ±0.5 bootstrap noise) and the rev2 10 × 10 seed grid.
- **L1**: add capture: weight the distance by survival, `⟨S(t) d(t)⟩` (rev2's R2), or raise τ
  so the capture-time term has gradient, still exactly zero-sum; measure the ensemble on the
  pre-capture window only.
- **L2**: the published protocol (3:1, warm-up, one-sided λ_nc, λ_d = 5) as the perturbation
  whose effect is then attributable.

"Is the simplest setup the exact zero-sum one?" Zero-sum is necessary but not sufficient:
C2 is exact zero-sum with three terms and a horizon that reaches heads. The simplest honest
game is a single zero-sum payoff that is itself the game of degree whose saddle is known in
closed form, and the paper's "disclosed exception" (shaping = value) is that game.

## 7. Forcing the latent not to collapse: the PR(H) regulariser (2 Nov 2026)

Question: does the Gram PR ≤ 2 reduction of the heading ensemble survive when the latent is
*not* allowed to reduce? Tool: `pursuit_evasion_numpy.pr_latent_reg`, an additive per-player
term `R_a = λ (n_b − PR(⟨H_a H_aᵀ⟩_t))` on the *uncentred* second moment of the body output
(gradient injected into `backward_dirs` via `dH_extra`; whole-loss finite-difference check
passes). Pilot flags: `--lam-pr λ [--pr-centred] [--pr-target]`. Runs: `viz/pilot_integ_tmax1.5_*lampr*`,
logs `viz/pilot_lampr_sweep.log`, `viz/pilot_lampr_long.log`. L0 pilot settings (t_max 1.5, 1:1, lr 3e-3).

Note on the name: a *sparsity* (L1) penalty would concentrate the spectrum, i.e. lower PR(H).
What is wanted is the opposite, a spectrum-flattening / anti-collapse term; PR is scale-invariant
and is the statistic we already report, so it is used directly. Whitening loss
‖n_b M/tr M − I‖²_F = n_b(n_b/PR − 1) is the same thing reparametrised.

Results (seed 0, 3000 iters unless stated; E/P):

| run | PR(H) centred | PR(H) uncentred | Gram PR | constancy η | J |
|---|---|---|---|---|---|
| λ = 0 (L0 pilot) | 6.97 / 7.38 | 3.12 / 4.07 | 2.01 / 2.02 | 0.993 / 0.988 | 0.7735 |
| λ = 0.003 | 14.1 / 14.3 | 14.9 / 15.1 | 2.60 / 2.70 | 0.836 / 0.818 | 0.7735 |
| λ = 0.01 … 1 | 14.1–14.4 | 15.0–15.2 | 2.3–2.7 | 0.78–0.89 | 0.760–0.775 |
| λ = 0.1, 10k iters | 14.3 / 14.3 | 15.1 / 15.1 | 2.15 / 2.16 | 0.929 / 0.916 | 0.7689 |
| λ = 1, 10k iters | 14.2 / 14.3 | 15.0 / 15.1 | 2.12 / 2.15 | 0.952 / 0.949 | 0.7604 |
| λ = 0.1 **centred** | 14.3 / 14.3 | 2.6 / 3.5 | 1.99 / 1.99 | 1.000 / 0.999 | 0.7652 |
| seeds 1, 2, λ = 0 | 6.5 / 6.5 | 2.8–3.9 | 1.95–1.99 | 0.997–0.998 | 0.751–0.757 |
| seeds 1, 2, λ = 0.1 | 14.2–14.5 | 15.0–15.3 | 2.18–2.45 | 0.80–0.94 | 0.751–0.766 |

Reading:
1. **The regulariser is almost free.** Already at λ = 0.003 the uncentred PR(H) saturates at
   ≈ 15 of 16, at no cost in J (the game value stays at the saddle-ish 0.77 ± 0.01). The network
   can make its latent isotropic without changing the strategies it plays.
2. **The rank-2 reduction of the strategy ensemble survives.** Gram PR goes 2.0 → 2.2–2.7 at
   3k iterations and comes back to 2.1–2.2 by 10k; η dips to 0.8–0.9 then recovers to 0.92–0.95
   and is still rising. So the reduction is *not* inherited from a collapsed latent.
3. **Mechanism: the heads go rank 1.** Under any λ > 0 the per-head singular-value ratio s₂/s₁
   drops from a median 0.34–0.49 (λ = 0) to 0.000 (max over heads ≤ 0.07). Each head becomes
   `z^j(t) = (w^j·H(t)) e^j`: a fixed direction e^j times a scalar, so all latent time variation
   is *radial* (99.9–100 % of Var_t z^j lies along the mean heading, vs 94–98 % at λ = 0) and is
   discarded by the normalisation `û = z/|z|`. The coefficient of variation of |z^j(t)| doubles
   (0.2 → 0.55–0.65). This is Lemma 2's gauge freedom used as a dump for the extra latent
   dimensions: the regulariser constrains `H`, the game constrains `û`, and the rank-1 heads
   make the two independent. The legacy PR(Z) ("tpr_Z") jumps to 2–5.5 for the same reason,
   confirming again that it is not the right statistic.
4. **The residual η < 1 is sign flips, nothing else.** For rank-1 heads `û^j = sign(w^j·H) e^j`;
   where `w^j·H(t)` crosses zero the heading flips by 180° for a short stretch. Per head the
   flipped time fraction is 1–10 % (15–16 of 16 heads at 3k), and η computed from these
   fractions alone, mean (1 − 2 f_j)², reproduces the measured η to three decimals (0.852 vs
   0.850, 0.781 vs 0.781, …). Off-axis headings (|cos| < 0.99) occupy only 1–2 % of the time.
   Training removes the flips slowly (η 0.78 → 0.95 by 10k): a flip is a stretch of maximally
   wrong play that the distance loss penalises, so this is the game winning the tension, as
   expected.
5. **The centred version is the predicted loophole.** With `--pr-centred` the body keeps a
   large constant mean plus tiny isotropic wiggles: centred PR(H) 14.3, uncentred 2.6–3.5,
   Gram PR 1.99, η 1.000. It tells us nothing. Regularise (and report) the uncentred moment.

Upshot for the paper: "PR(H) small" and "Gram PR = 2" are logically independent in this
architecture. The ensemble reduction is a statement about the heads and the game, and a
latent of any PR can carry it, because linear heads followed by a normalisation can route an
arbitrary number of latent directions into the discarded radial component. A latent PR
measurement therefore cannot by itself support the "the embedding reduces to the game's
dimension" reading; the gauge-fixed head rank (`erank_gf`) and the Gram PR are the objects.

Hinge-squared form (added 3 Nov 2026, `--pr-form hinge2 --pr-target 12`, R = λ max(0, PR* − PR)²): the
term parks the uncentred PR(H) at the target and then switches off. Seed 0, λ = 0.1: Gram PR
1.98/2.05, η 0.997/0.975 at 3k; 2.01/2.02, η 0.990/0.985 at 10k, PR(H)unc 13.3/12.8. Seeds 1–2 the
same (Gram PR 2.04–2.15, η 0.95–0.98). Executed analysis notebook with all figures:
`viz/analysis_pr_regulariser.ipynb` (figures saved to `viz/pr_reg_analysis/`).

Cleaner control still to run: bypass the body and feed the heads `φ(t)` directly (H = φ,
exactly isotropic with PR = 2K + 1 by construction, no λ to tune) and check Gram PR there.
