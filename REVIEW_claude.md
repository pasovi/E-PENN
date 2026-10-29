# Review of "What Does a Game-Trained Embedding Learn?" (Jimenez et al., draft of 8 Oct 2026)

Reviewed against the reproduction bundle in this folder. Every number below was recomputed from the bundle unless marked otherwise. A copy of the bundle was run in a scratch directory; nothing in this folder was modified except the addition of this file.

---

## 1. What the paper says (summary)

**Setup.** A multi-head network from Tarancón-Álvarez et al. (2026) is trained by adversarial self-play on an open-loop pursuit–evasion game on the flat torus (pursuer speed 1.2, evader 1.0, horizon 4, capture radius 0.08). Each player has a shared body (Fourier features of time with K = 12 harmonics, one tanh layer, 16-dim latent H(t)) and 16 linear heads, one per evader initial condition, each outputting a unit heading. Losses are a differentiable soft capture time plus a no-capture penalty (pursuer only) and a distance-shaping term. Nothing analytic enters any loss ("honesty contract"), except that on the torus the distance-shaping term is provably an affine image of the analytic value, which the paper discloses and measures.

**Question.** What does the 16-dim latent H(t) encode? The paper's answer is that two different objects must be separated.

**Main results.**
1. **Strategy ensemble is low-rank.** The 16 control curves have participation ratio (PR) ≈ 2–3 (baseline 2.99 evader, 1.91 pursuer), which the paper reads as the constant-heading saddle prediction of 2 plus soft-game corrections. A warm-start control (C7) shows the architecture can fit a rank-9 curved family and game training then collapses it to rank 2–3, so the low rank is selected, not built in. A shaping sweep (C5) shows it does not depend on the disclosed shaping term.
2. **Latent curve carries an overhead that no head uses.** PR(H_E) ≈ 5.4 ± 0.9 over init seeds. Three controls attribute it to the input featurisation: the latent spectrum breaks at the input bandwidth K and the break moves with K (C3); the latent PR is unrelated to the number of winding classes realised (8 runs, C4); and its variance is dominated by the random init rather than the IC sample (C1, 4.7:1). Latent PR exceeds strategy PR by at least 1.69× in every one of 43 checkpoints.
3. **Topology appears as clusters, not dimensions.** Winding classes of evader trajectories (signed, Z²) are visible as tight clusters in control space (within-class L² ≈ 0.02–0.07 for four of five pairs; cross-class ≈ 1.25) but cannot be seen by any spectral statistic (Proposition 1: constant headings span a 2-dim subspace regardless of homotopy class). A reflection control confirms the winding is a genuine signed class.
4. **Parameter sweep.** Across the speed difference Δv ∈ [0.05, 0.5], which carries the game through its capturability transition at ρ = 1, strategy PR stays 1.4–3.1 and latent PR stays 5–7 with no feature at the transition.
5. **A tempting asymmetry is withheld.** Raw head singular values differ 3× between players, but a gauge lemma shows these are not observables; after gauge fixing the ordering inverts. A gauge-invariant 2× pursuer excess of intermediate-band power survives seeds and shaping annealing, but under a fully symmetrised protocol (C2) it becomes seed-dependent and straddles the pre-registered 1.5× threshold, so no game-theoretic attribution is made.
6. **Training dynamics (Section 8).** The equilibrium is mediocre (23% capture of the capturable set) because of co-adaptation, not capacity: against a fixed optimal evader the pursuer converges fully; simultaneous GDA and alternating best response cycle; none of optimistic GDA, extragradient, Cesàro averaging or EMA reaches the oracle basin. Grouped Shapley over input frequency bands shows the oracle's capture lives in the k = 1–4 band, which the self-play pursuer never acquires.
7. **Value-level check (Section 9)** against an independently trained HJI critic from a companion paper: actor–critic alignment factorises through the analytic saddle direction; no latent-to-value bridge exists (latent components predict the critic's value no better than time alone).

---

## 2. Does the code reproduce the paper?

**Short answer: the numbers yes, the figures mostly yes, with gaps in what is shipped.**

| Paper element | Reproduced? | How verified |
|---|---|---|
| Fig. 1 (integer-k latent spectra) | **Content yes, script missing.** No script in the bundle draws this figure; `embedding_analysis.py` makes a different (log-shell) plot. Reconstructed from `v2_actor.npz` using the conventions in `revision/post_hoc_checks.py`; the curves, the 27×/9× drops at k = 12→16, the dashed K line and the shaded band match the committed PNG exactly. | reconstruction + visual comparison |
| Fig. 2 (sectors) | Yes. `make_fig_sectors.py` regenerates the same data (PR 6.7/5.5/5.0 vs 3.0/2.8/2.2; 12 and 3 classes). Cosmetic differences only (the committed PNG was rendered in a separate pass). | run + visual |
| Fig. 3 (reflection) | Yes. `make_fig_winding_reflection.py`; mean w_x +1.05 → −0.86. Cosmetic differences only. | run + visual |
| Fig. 4 (Δv sweep) | Yes, pixel-identical. | run + pixel diff |
| Fig. 5 (head SVs raw vs gauge-fixed) | **Content yes, script missing.** `embedding_analysis.py` draws only the raw panel and no init line. Reconstructed from `v2_actor.npz` + `init_params(seed 0)`; max SVs 18.07/5.58, gauge-fixed PR 2.53/1.91 and the init curve match the committed PNG exactly. | reconstruction + visual |
| Fig. 6 (protocols), Fig. 7 (honest protocols), Fig. 8 (Shapley) | Yes. Regenerated images differ from the committed PNGs only by a few pixel rows (font metrics of a different matplotlib); the summary JSONs are byte-identical. | run + diff |
| Table 1 numbers (C1–C7) | Yes. `revision/analyze_e1e7.py` regenerates `e1e7_analysis.json` identically; C1/C2/C3/C4/C5/C7 numbers all trace to `revision/*.json` and were recomputed from the `.npz` checkpoints. | run + recomputation |
| Section 3 "rank two under stronger play" (1.7 / 2.1 / 2.4–2.7) | Yes (`oracle_pr.py`, `pedro_oracle_pr_result.json`). | run |
| Section 5 winding distances (0.02–0.07, 0.81, 6×, ≈20×) | Yes. | recomputation |
| Section 7 band fractions (30.3% vs 15.8%; 22.7% vs 11.5%; ×1.44–1.92; C2 ×1.25/1.51/1.64/1.99; [0.66, 3.29]) | Yes, but see §3.3 below on which runs are counted. | recomputation |
| Section 9 numbers (31%/23%, 0.433/0.541, R² 0.46) | Yes from stored JSON; the 0.53 "time alone" figure is not in any stored file and was not recomputed. | JSON |
| Section 2 shaping shares (≈100% / 6.4%; ×45) | Yes (`post_hoc_checks.json` G; `e1e7_analysis.json`). | JSON |

**Bundle issues worth fixing before release**
- The README says the four figure scripts match the committed PNGs byte-for-byte. They match in data, not in bytes, across matplotlib versions. Say "data-identical".
- Ship plotting scripts for Figures 1 and 5 (they are the two figures that carry the featurisation and gauge arguments).
- The README says training needs torch and that the critic code was removed. Neither is true: all training scripts are pure NumPy with hand-coded gradients, and `pursuit_evasion_v4.py`, `run_v4.py`, `run_v5.py`, `honest_critic_geom.py`, `run_geom.py`, `geometries.py` and the `v4_honestV.npz`/`v5_honestV.npz` critics are present (and `revision/post_hoc_checks2.py` needs them).
- The dv-sweep, protocol and Shapley experiments ship only their result logs (`*.jsonl`, `shap_seed*.json`), not checkpoints. Figures 4, 6, 7, 8 therefore reproduce from logs; regenerating the logs requires retraining (24 + 12 + 18 + 6 runs).
- `revision/post_hoc_checks.json` still contains a `KeyError('W1')` in its `H_bridge` section (fixed in `post_hoc_checks2.py`); the "12,000-iteration check" quoted in the Fig. 7 caption is not in the bundle.
- `paper/` contains 16 PNGs not referenced by the TeX (companion-paper and exploratory figures).
- The analysis scripts overwrite the summary JSONs in place when run; harmless but worth noting.
- The author list in the TeX lacks the "-Pérez" that the 8 Oct PDF has; otherwise the TeX and PDF text are identical.

---

## 3. Correctness problems found in the text

These are the items I would expect a referee to catch. Items 3.1 and 3.2 affect claims in the abstract.

### 3.1 The "population range [1.46, 2.99]" and "the two populations do not overlap" are not true as stated
Section 3 says the strategy-PR range across "eight init seeds, five IC sets, the sector restrictions, the shaping sweep, the warm-start run, and twenty-one symmetric-protocol snapshots" is [1.46, 2.99], and that the latent and strategy populations do not overlap. Recomputed over the same 43 checkpoints listed in `revision/within_run_ratios.json`:

| statistic | paper | recomputed |
|---|---|---|
| PR(Z_E) range | [1.46, 2.99] | [1.46, 3.61] |
| PR(Z_P) range | (not given) | [1.31, 4.30] |
| PR(H_E) range | [3.65, 6.75] (8 seeds) | [3.65, 8.04] (43 checkpoints) |
| PR(H_P) range | (not given) | [4.83, 8.38] |

Nine checkpoints exceed 2.99: six C2 snapshots (up to 3.61 evader, 4.30 pursuer) and both E7 snapshots (3.33, 3.45). [1.46, 2.99] is the evader-only range over the non-symmetric, non-annealed runs. Since the pursuer's PR(Z) reaches 4.30 and the evader's PR(H) goes down to 3.65, the populations do overlap. The within-run statement (ratio ≥ 1.69 in every checkpoint, both players) is correct and is the one to lead with. The abstract's "1.5–3.0" needs the same qualification.

### 3.2 Two different participation ratios are used under one name, and Proposition 1 bounds the other one
The headline PR(Z) (2.99 / 1.91, Figs. 2 and 4, C5, C7) is the PR of the **temporal covariance** of the 32-dim stacked head outputs, centred over time. The "rank two under stronger play" paragraph (saddle ≈ 1.7, actor ≈ 2.1, oracle ≈ 2.4–2.7) uses `oracle_pr.py`'s `head_PR`, which is the PR of the **Gram matrix across the 16 heads** of the unit-heading curves as vectors in L². These are different statistics:

| object | temporal PR | across-head Gram PR |
|---|---|---|
| baseline Z_E | 2.99 | 2.37 |
| baseline Z_P | 1.91 | 2.33 |
| baseline unit headings, evader | 2.85 | 2.70 |
| baseline unit headings, pursuer | 3.14 | 2.62 |
| analytic constant-heading saddle | undefined (zero temporal variance) | 1.94 |
| C7 planted curved family | 9.26 | 10.22 |

Proposition 1 ("constant R²-valued curves span a two-dimensional subspace") bounds the across-head Gram PR. It says nothing about the temporal PR, which for the ideal saddle is 0/0 and for perfectly constant headings with time-varying gains can be anything up to 16. So the sentence "Section 2 predicts PR(Z) ≈ 2 plus soft-game corrections. We measure PR(Z_E) = 2.99" and the dotted "constant-heading bound (= 2)" in Figs. 2 and 4 attach a theorem to the wrong statistic. The empirical result (temporal PR ≈ 2–3, learned not default, per C7) is still interesting; the mechanism is the rank chain through the stacked head matrix (gauge-fixed effective rank 2.5 / 1.9), not Proposition 1. Fix: define both PRs explicitly, report the across-head PR as the quantity Proposition 1 bounds, and describe the temporal PR as "the number of temporal modes the heads read from H", justified by the rank chain.

### 3.3 Seed and run accounting does not add up
- **C1.** "PR(H_E) = 5.4 ± 0.9 (n = 8 seeds; range [3.65, 6.75])". The mean 5.4 ± 0.9 is over the seven non-baseline seeds in `revision/c1_completion.json` (range [3.65, 6.28]). Including the baseline seed 0 (6.75), which the quoted range does, gives 5.6 ± 0.9 (n = 8). The variance ratio 4.7:1 is also the 7-seed value (8-seed: 5.2:1). Pick one n and use it everywhere (abstract, Sec. 3, Sec. 4, Table 1).
- **C2.** The paper reports four seeds with run-mean gaps ×1.25, 1.51, 1.64, 1.99 and a snapshot range [0.66, 3.29]. The bundle contains **eight** C2 runs (41 snapshots): the original `c2_*` run (6 snapshots, mean 1.02, three snapshots below 1, including the 0.66 that enters the quoted range), a second seed-0 run `c2_s0_*` (mean 1.25, the value quoted), and seeds 1–6. Seeds 4–6 (means 1.53, 1.20, 2.40) appear nowhere in the paper or in `revision/*.json`. Over all eight runs the mean gap is 1.55, range [0.66, 3.46], 20% of snapshots below 1, i.e. the "withhold" verdict is unchanged, but the text must say that two seed-0 runs exist (different Adam-reset cadence), which one is which, and report all seeds. As written, the 0.66 comes from a run whose mean (1.02) is not in the list of means.
- **"43 training runs".** The abstract and Sec. 4 say 43 training runs; Sec. 3 correctly says 43 checkpoints. They come from 25 distinct runs (21 of the 43 are C2 snapshots from four runs; E7 contributes two snapshots of one run).
- **"cross-class median 1.25"** (Sec. 5) is the cross-class mean; the median is 1.29. "6× separation" is mean/mean (6.2); median/median is 20.

### 3.4 Claims that are weaker than their wording
- "Late-training oscillation grows monotonically with the symmetry of the co-adaptation" (Fig. 6): self-play 3:1 has per-seed oscillation 0.0034 / 0.037 / 0.0037 and simultaneous 1:1 has 0.0095 / 0.040 / 0.0068. The means (0.015 vs 0.019) differ by less than the seed spread. Only oracle ≪ (3:1 ≈ 1:1) ≪ ABR is supported.
- The same protocol (simultaneous 1:1 GDA, same learning rates and ICs) has oscillation 0.019 in Fig. 6 (14,000 iterations) and 0.077 in Fig. 7 (6,000 iterations). The two figures should state budgets and comment on the 4× difference, or use one budget.
- "No relation to class count" (Sec. 4, Fig. 2 left): within Fig. 2 both PR(H_E) (6.7 → 5.5 → 5.0) and PR(Z_E) (3.0 → 2.8 → 2.2) fall monotonically with class count (12 → 5 → 3). The defensible statement is that the sector effect is smaller than the seed scatter (3.65–6.75) and that the capturability-matched C4 (4 classes, 5.22) sits at the seed mean. Say that; do not say "no relation" on n = 3 monotone points.
- C7 "within 8,000 iterations": PR(Z_E) is 3.01 at 8k and reaches 2.46 only at 14k (first below 3 at 3k, then fluctuates). "Collapses to 2–3 within 8k" is fine; "6.5 → 2.5 within 8k" is not.
- Fig. 4b plots only the evader's latent PR. The pursuer's (5.2–8.1 across Δv) is in `dv_sweep_summary.json`; either show both or say why not.
- The "≈100%" evader shaping share is 100.7% (vector norms do not add), which is fine but should be stated as "the shaping gradient alone has the same norm as the full gradient".

---

## 4. Clarity and presentation

**What works.** Definitions are explicit; every claim has a named control with a pre-stated two-sided rule (Table 1); nulls and withheld attributions are reported with the same prominence as positives; the gauge lemma and the shaping-is-value disclosure are exactly the kind of thing referees want and rarely get.

**What does not.**
1. **Density.** The abstract contains eleven numbers and the introduction's four "findings" paragraphs are results sections in miniature. A reader meets PR = 5.4 ± 0.9, 4.7:1, 43 runs, 6×, ×1.25–1.99 before learning what the game or the network is. Move every number out of the abstract except one or two, state findings qualitatively in the introduction, and let Sections 3–7 carry the numbers (preferably in one results table beside Table 1).
2. **Jargon introduced faster than it is defined.** "Featurisation residue", "honesty contract", "gauge", "actor/critic" (with a disclaimer that it is not RL actor–critic), "capturability transition", "inf–sup structure", "hopeless heads". Each is fine once defined; several are used before definition.
3. **Dependence on an unseen companion paper.** Section 9 cannot be evaluated by a reader of this paper alone, and "Companion paper (2026)" is cited eight times. Either make Section 9 self-contained (one paragraph on how the critic is trained and validated) or cut it to a short remark and keep the null for the latent-to-value bridge.
4. **No picture of the problem.** There is no figure of the torus game, no architecture schematic, and no example trajectories, although `paper/fig_traj_actor.png` exists. A Figure 0 with (a) the game and the capturable disk, (b) the body/heads network, (c) three trajectories would halve the cognitive load of Section 2.
5. **Two papers in one.** Sections 8 (training-protocol comparison, honest corrections, Shapley) are a self-contained study of self-play optimisation that answers a referee question rather than the title question. They dilute the embedding story and add four of the eight figures. Consider an appendix or a separate short note, keeping only the oracle-vs-self-play rank comparison and the one-sentence Shapley corroboration in the main text.
6. **Missing scaffolding.** The proofs of Proposition 1, Lemma 2 and the shaping identity are "in the supplementary repository" but are not in the bundle; the pre-registered designs and decision rules exist only as script docstrings. Add an appendix with the three proofs (each is a few lines) and the committed decision rules verbatim.
7. **Related work is one paragraph and eleven references** for a paper that touches representation analysis, spectral bias, multi-head PINNs, differential-game solvers and min–max optimisation. See §6.
8. **Sentence-level.** Many sentences carry three clauses and two parentheticals ("The PR over 16 curves carries a bootstrap uncertainty of about half a unit, so ranges rather than decimals carry the conclusions" is good; "Over all 16 heads the oracle's rank inflates to ≈ 7, but only because its uncapturable heads chase evaders they cannot reach, so their headings are undetermined; this is an artifact of the free heads" is three sentences). The Limitations section is mostly a plan for five future papers; cut it to the actual limits (one game, one geometry, open-loop t-only policies, 16 heads, a 23%-capture equilibrium, n = 3–8 seeds).

**Figures, specific.**
- Fig. 1: add the K = 6 and K = 24 spectra (C3) as a second panel or inset. The "break moves with K" result is the paper's central featurisation evidence and currently has no figure.
- Fig. 2 right: title promises "integer cells" but no cell boundaries are drawn; add grid lines at half-integers and label a few cells. Legend overlaps data at bottom-right. Left: label the x-axis categories by what changed ("±15° sector, r ≤ 1.35" etc.), not "E1 angular".
- Fig. 3: mostly white space; fold into Fig. 2 as a third panel.
- Fig. 4b: add the pursuer; mark K-set scale (≈5–7) rather than only the "=2" line.
- Fig. 5: fine. Consider adding gauge-fixed PR values in the panel.
- Fig. 6a: legend hides the ABR trace; the s.d. band is clipped at 10⁻³ on a log axis, which visually understates the self-play spread; the (b) title is a sentence.
- Fig. 7a: the legend box sits on top of the region where the oracle basin annotation lives; small per-seed markers are hard to read; (b) shows four of six protocols without saying so.
- Fig. 8: fine; move the annotation text to the caption.
- Table 1: add a column with the pre-registered decision rule in one line per control, and a column with the figure/JSON that holds the result.

---

## 5. Smaller technical points
- Open-loop time-only strategies with a fixed origin for the pursuer make each head a 1-D curve-fitting problem; the "game" couples heads only through the shared body. State this plainly up front; it also explains why the strategy rank is tiny.
- The model is very small (body 16 × 25, heads 16 × 2 × 16). A sentence on why this scale is the right test-bed (it is the Burgers architecture at the same width) would pre-empt the "toy" objection.
- The capturable set is 4 of 16 heads at the baseline. Several headline numbers (rank on capturable heads, Shapley) rest on four curves; say so where they are used, as the paper already does once.
- `winding_class_flip_frac_pm0.1` = 0.625 in `e1e7_analysis.json` (62% of heads change integer class if the rounding threshold moves by ±0.1) is a relevant robustness number for Section 5 that the paper does not report. It argues for showing the continuous winding cloud (as Fig. 2 does) rather than integer class counts.
- The Δv sweep's "three seeds" have latent-PR s.d. up to 1.3; Fig. 4b's error bars are population s.d. over three values. Say "range over three seeds".

---

## 6. Novelty and whether this is enough for a paper

(Literature search results are appended below in §7.)

**Genuinely new, as far as I can tell.** (i) An explicit audit that separates the rank of the strategy ensemble from the rank of the shared latent in a multi-head game-trained network, with controls that attribute the latent overhead to featurisation (K-sweep), initialisation (seed vs IC variance split) and not topology (C4). (ii) The gauge lemma for per-head rescaling applied to head-spectrum readings, and the demonstration that a gauge-invariant, seed-stable, theory-consistent asymmetry still fails a pre-registered symmetrisation control. (iii) The observation that homotopy content is visible as clusters in control space but invisible to spectral statistics (Proposition 1 is trivial, but the point is useful). (iv) The grouped-Shapley-over-frequency-bands diagnostic for an oracle gap.

**Known results re-derived.** The latent spectrum breaking at the input Fourier bandwidth is what Fourier-feature and spectral-bias theory predicts (Tancik et al. 2020; Rahaman et al. 2019; Wang, Wang and Perdikaris 2021); the paper should cite this and present C3 as a confirmation rather than a discovery. Simultaneous GDA cycling and the failure of convex–concave corrections on nonconvex–nonconcave games are well documented (Mescheder et al. 2018; Daskalakis et al. 2018; and the fictitious-play / PSRO / league-training line). Constant-heading open-loop saddles on the flat torus are classical.

**Is it enough?** As a focused interpretability/audit paper the core (Sections 2–5, 7, with C1–C5, C7) is a coherent, carefully controlled contribution with an honest negative result. It is not, in its current form, a strong fit for a general ML venue: one game, one geometry, open-loop time-only policies, 16 initial conditions, a tiny network, a mediocre equilibrium, n = 3–8 seeds, and a headline ("strategies are rank 2") that follows almost directly from the open-loop constant-heading structure. Three things would change that assessment: (a) a closed-loop state-conditioned actor (the paper itself names this as the only extension that could alter the ensemble-rank story); (b) a second geometry (disk or Möbius strip, for which the bundle already contains critic code and figures) where shaping is not value-equivalent; (c) tightening the claims per §3 so that the two PR definitions, the population ranges and the seed counts are exact. With (c) alone, the paper is a defensible workshop or TMLR-style submission framed as a methodology-plus-negative-result paper; with (a) or (b) it becomes a full paper.

---

## 7. Literature search (web, 8 Oct 2026)

Found by a web search run for this review. The pre-2026 items are ones I recognise as correct; the 2026-dated arXiv numbers were reported as verified on arXiv but should be double-checked before citing.

**7.1 Neural / PINN solvers for differential and pursuit–evasion games (absent from the paper).** Zhang, Ghimire, W. Zhang, Xu, Ren, "Value approximation for two-player general-sum differential games with state constraints", IEEE T-RO 2024, arXiv:2311.16520; Zhang et al., "Pontryagin neural operator", L4DC 2024, arXiv:2401.01502; Zhang et al. 2025 arXiv:2503.06994 and Ghimire, Xu, Ren 2025 arXiv:2502.05314 (parametric and one-sided-information variants); Bokanowski & Warin 2024 arXiv:2402.02792 (NN feedback strategies with error estimates); Ly & Tsai arXiv:2010.09001 (surveillance–evasion, HJI + self-play); Grupen, Lee, Selman 2021 arXiv:2106.11156 (multi-agent pursuit–evasion **with periodic boundary conditions**, i.e. a torus, decentralised DDPG); DeepReach follow-ups (Lin & Bansal 2022; Hofgard 2024 arXiv:2410.02904). None audits representations or uses a shared-body/heads architecture; the ASU line and Grupen et al. should be cited.

**7.2 Interpretability of self-play / RL representations.** McGrath et al., PNAS 2022, arXiv:2111.09259 (AlphaZero concept probing); Lovering et al., NeurIPS 2022, arXiv:2211.14673 (AlphaZero-Hex probing); Zabounidis et al., CoRL 2022 (concept-bottleneck MARL); Chen et al. 2019 arXiv:1910.07882 (visual hide-and-seek representations). Dimensionality of RL features: Kumar, Agarwal, Ghosh, Levine, ICLR 2021, arXiv:2010.14498 (effective-rank collapse); Lyle, Rowland, Dabney, ICLR 2022, arXiv:2204.09560; Gulcehre et al., TMLR 2022, arXiv:2207.02099. So "representation audits of game-trained networks" do exist; the paper's claim should be restricted to physics-informed / differential-game architectures.

**7.3 Participation ratio and intrinsic dimension; init-set dimensionality.** Gao et al. 2017 bioRxiv 214262 (PR as the standard dimensionality measure); Litwin-Kumar et al., Neuron 2017; Recanatesi et al. 2019 arXiv:1906.00443; Ansuini et al., NeurIPS 2019, arXiv:1905.12784; Flesch et al., Neuron 2022 (rich vs lazy regimes: initial weight scale sets representation geometry, directly relevant to "variance is set by the initialisation"); Chun et al. 2025 arXiv:2509.26560 (finite-sample bias of PR, relevant since PR is computed from one 16-dim curve). None of these are cited.

**7.4 Spectral bias / Fourier features.** Rahaman et al., ICML 2019, arXiv:1806.08734; Tancik et al., NeurIPS 2020, arXiv:2006.10739 (the feature frequencies set the learnable band); Wang, Wang, Perdikaris, CMAME 2021, arXiv:2012.10047 (eigenvector bias of Fourier-feature networks). The "break at K" (C3) is a direct consequence of these and should be presented as such.

**7.5 Multi-head / shared-body PINNs and latent analysis.** Zou & Karniadakis 2023 arXiv:2301.02152 (L-HYDRA, shared body + linear heads); Desai et al. 2021 arXiv:2110.11286 (one-shot transfer via linear heads); Penwarden et al., JCP 2023, arXiv:2110.13361; HyperPINN arXiv:2111.01008; Cho et al., ICML 2024 (P²INNs); Seiler, Lei, Protopapas 2025 arXiv:2501.17281; DeepONet (Lu et al., Nat. Mach. Intell. 2021) as the structurally identical trunk-basis × linear-coefficients decomposition. **Most important:** the search reports a follow-up by the same group, Jimenez, Mayboroda, Protopapas, Sarieddine, Spergel, Tarancón-Álvarez 2026, arXiv:2607.06348, which already performs PCA and Fourier-shell analysis of the same latent architecture on PDE families. If that is right, this manuscript must cite it and state exactly what is new relative to it (the game-trained setting and the strategy-vs-latent split).

**7.6 Self-play / min–max optimisation.** Heinrich & Silver 2016 (NFSP, arXiv:1603.01121); Lanctot et al., NeurIPS 2017 (PSRO, arXiv:1711.00832); Vinyals et al., Nature 2019 (league training); Daskalakis & Panageas, NeurIPS 2018, arXiv:1807.03907; Mertikopoulos et al., ICLR 2019, arXiv:1807.02629; Gidel et al., ICLR 2019, arXiv:1802.10551; Yazıcı et al., ICLR 2019, arXiv:1806.04498 (iterate/EMA averaging); Mazumdar, Ratliff, Sastry, SIMODS 2020; Schäfer & Anandkumar, NeurIPS 2019 (CGD). Section 8's conclusions are consistent with this literature and should be framed as a confirmation on a new game, with these citations.

**7.7 Homotopy / winding in planning and pursuit–evasion.** Bhattacharya, Likhachev, Kumar, Auton. Robots 2012 (homotopy classes via complex-analytic invariants); Pokorny, Hawasly, Ramamoorthy, RSS 2014 / IJRR 2016 (persistent-homology trajectory classes); Pokorny, Goldberg, Kragic, ICRA 2016 (topological trajectory clustering with winding centres). No prior work classifies learned pursuit–evasion strategies on T² by net winding; this part appears new, but the trajectory-clustering literature is the right context.

**7.8 Shapley over input groups / frequency bands.** Jullum, Redelmeier, Aas 2021 arXiv:2106.12228 (groupShapley); Luo, McDermott, O'Riordan, TMLR 2024, arXiv:2408.01139 (Shapley over spectral bands of inputs, the direct precedent); grouped time-series Shapley (ShaTS 2025 arXiv:2506.01450); SVERL for RL (Beechey, Smith, Şimşek, arXiv:2505.07797). Applying it to Fourier bands of a controller's time input is a new application, not a new method.

**Net novelty assessment.** New: the audit of a game-trained multi-head physics-informed embedding that separates strategy-ensemble rank from latent rank; attribution of the latent surplus to featurisation and initialisation with a pre-registered control set; reading topology off per-head net winding on the torus; the gauge-lemma-plus-symmetrisation negative result. Known and re-derived: bandwidth-bounded spectrum (Tancik; Wang–Wang–Perdikaris), GDA cycling and its standard fixes (Daskalakis; Mertikopoulos; Gidel; Yazıcı), init-set dimensionality in the lazy regime (Flesch; Litwin-Kumar), PR/ID tooling. The related-work section needs roughly 25 additional references across 7.1–7.8, and the group's own arXiv:2607.06348 must be distinguished explicitly.
