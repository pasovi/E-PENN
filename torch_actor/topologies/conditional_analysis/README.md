# `conditional_analysis/` — is the TL improvement with d a dimension effect or a sampling effect?

The transfer-learning success on the capturable new ICs rises with the dimension (S¹ 76 %, T² 91 %, T³ 97 %,
T⁴ 98 %) while, at fixed reach R = 0.88 and box L = 2, the capturable set is a shrinking share of the sampled
ICs. `conditional.py` asks whether the error rate still depends on d once the *position* of each IC inside the
capturable ball is controlled for. No new training: it reads the four existing TL runs
(`../results/{S1,T3,T4}_tl.json`, `../../transfer_learning/results/D1_grid1024_seed0.json` for T²).

Two geometric coordinates per capturable IC: the reach margin b = R − |Δ0| (small b: outer shell of the
capturable ball) and the image margin m = d₁ − d₀, the distance to the second-nearest periodic image of the
pursuer minus the distance to the nearest (small m: a competing image, i.e. the IC sits near a cut-locus wall).
The TL errors on T^d, d ≥ 2, are heads that chase a non-nearest image, so they should be a function of (b, m).

```bash
python3 torch_actor/topologies/conditional_analysis/conditional.py
```
Outputs: `results.json` (every count, the cells, the fits), `summary.md` (the tables below),
`figs/conditional.pdf` (top: (b, m) scatter of the capturable ICs per dimension with the escaped ones ringed;
bottom: error rate against b and against m, the danger-zone share under uniform sampling against what each
run's sampler actually drew, and the geometry-standardised rates).

## Results (6 Nov 2026)

| | S¹ | T² | T³ | T⁴ |
|---|---|---|---|---|
| TL sampler | grid 100 | grid 32² | grid 10³ | Sobol 1000 |
| capturable ICs / errors | 88 / 21 | 616 / 56 | 360 / 10 | 179 / 3 |
| error rate (95 % CI) | 23.9 % [16.2, 33.7] | 9.1 % [7.1, 11.6] | 2.8 % [1.5, 5.0] | 1.7 % [0.6, 4.8] |
| capturable fraction of ICs at |Δ0| ≥ 0.4, uniform sampling | 47.9 % | 48.5 % | 32.5 % | 17.8 % |
| danger-zone share (b < 0.1, m < 0.5) of the capturable ICs, uniform | 20.8 % | 16.8 % | 12.2 % | 8.5 % |
| danger-zone share, actual sample | 11.4 % | 13.0 % | **0.0 %** | 6.1 % |
| errors / ICs inside the zone | 5 / 10 | 27 / 80 | 0 / 0 | 3 / 11 |
| errors / ICs outside the zone | 16 / 78 | 29 / 536 | 10 / 360 | 0 / 168 |

Pooled logistic regression over T², T³, T⁴ (1,155 capturable ICs, 69 errors): logit p = 5.4 + 25.3 b − 18.5 m.
Geometry against the null model: LR = 183 (2 df), p = 2 × 10⁻⁴⁰. A per-dimension intercept on top of the
geometry: LR = 1.07 (2 df), **p = 0.59** (odds ratios relative to T²: T³ 0.75, T⁴ 0.58, both compatible with 1).

| | observed errors | expected from the geometry-only model | observed rate | geometry-standardised rate |
|---|---|---|---|---|
| T² | 56 | 53.3 | 9.1 % | 8.7 % |
| T³ | 10 | 11.4 | 2.8 % | 3.2 % |
| T⁴ | 3 | 4.3 | 1.7 % | 2.4 % |

## Reading

**The image margin m is the predictor, and its effect is the same in every dimension.** The error rate
against m (bottom middle of the figure) is one curve for T², T³ and T⁴: about 40–60 % of the capturable ICs
with m < 0.4 escape, 10–30 % with m in [0.4, 0.6), essentially none with m > 0.8 (3 errors in 1,000 ICs
there). Conditioned on (b, m), the dimension adds nothing (p = 0.59), and the geometry-only model reproduces
each run's error count (53 vs 56, 11 vs 10, 4 vs 3).

**So the improvement with d is a sampling effect, in two parts.** (i) The share of the capturable set that has
a competing image shrinks with d under uniform sampling (danger zone 17 % → 12 % → 8.5 % for d = 2, 3, 4):
the boundary of the capturable ball comes close to a cut-locus wall only along the axes, and the solid angle
of those axis directions falls with d. (ii) The samplers add their own bias: the 10³ cell-centred grid on T³
has *no* IC in the danger zone (its outer-shell points, |Δ0| ∈ [0.78, 0.88], all have m ≥ 0.5), so the T³
figure of 97 % is flattered; the 32² grid on T² over-samples it slightly (13 % vs 17 % uniform) only in the
sense of quantising m on a few values; the Sobol set on T⁴ sits a little under the uniform share (6 % vs 8.5 %).

**Consequence for the Δv-matched control.** It is not needed to settle this question: equalising the
capturable *volume* would not equalise the danger-zone share, which is what the error rate tracks, and the
conditional rates already coincide. The honest cross-dimension comparison is the error rate at fixed m, or
the standardised rate above. What would still carry information is a per-dimension TL run on ICs sampled
uniformly *inside the capturable ball* (same number of capturable heads and the same m distribution in
every d), which removes both sampling biases at once.

**S¹ is a different mechanism** and is excluded from the pooled fit: its errors (21 of 88) are heads running
the wrong way round the circle, spread over all b (b from 0.03 to 0.63), and m = L − 2|Δ0| carries no extra
information there. Its 24 % error rate is not on the T^d curve.
