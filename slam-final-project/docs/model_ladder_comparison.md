# Model ladder comparison — es_en native task (all evaluated on the official es_en dev)

Every model fit on es_en train, evaluated on the real es_en dev (288,864 tokens,
2,640 learners) — the same dev set as the official pipeline, so AUROCs are directly
comparable to the project's official hierarchical result (0.7554). MAP (L2 logistic)
analogues throughout; each block corresponds to a hierarchical Bayesian prior and
ports cleanly to PyMC.

## Headline table

| # | Model | AUROC | Brier | NLL | ECE | Δ vs previous (clustered 95% CI) |
|---|---|---|---|---|---|---|
| M1 | Pooled GLM | 0.6700 | 0.1261 | 0.4103 | 0.0097 | — |
| M2 | + learner intercepts (≈ current final model) | 0.7144 | 0.1220 | 0.3962 | 0.0100 | — |
| M3 | + **word/item difficulty (Bayesian IRT)** | 0.7554 | 0.1154 | 0.3767 | 0.0071 | **+0.041** [+0.039, +0.043] |
| M4 | + **dynamic ability** (decayed past-error, leakage-safe) | 0.7611 | 0.1144 | 0.3736 | 0.0069 | +0.006 [+0.004, +0.008] |
| M5 | + **memory/HLR features** (exposures, per-word error history, lag) | 0.7709 | 0.1129 | 0.3689 | 0.0078 | +0.010 [+0.009, +0.011] |
| M6 | + **learner×format random slopes** (shrunk) | **0.7800** | 0.1105 | 0.3620 | 0.0081 | +0.009 [+0.008, +0.011] |
| M7 | GBM, same info set (BART-style nonlinearity check) | 0.7692 | 0.1132 | 0.3682 | 0.0104 | M6 − M7 = **+0.011** [+0.007, +0.013] |
| — | **Stacked ensemble** (all 7, logit stack; learner-split test half) | **0.7937** | — | — | — | +0.013 over M6 [+0.010, +0.016] |

Every rung is individually significant under the learner-clustered bootstrap.
Reference points on this same dev set: official project hierarchical model 0.7554;
2018 shared-task top systems ≈ 0.86.

## Conformal selective prediction (learner-split cal/test)

Split-conformal abstention with finite-sample risk control — the
distribution-free complement to the Bayesian routing layer:

| Model | Target risk | Coverage | Empirical test risk |
|---|---|---|---|
| M6 (best single) | 5% | 45.6% | 5.05% |
| M6 | 10% | 82.0% | 9.86% |
| Stack | 5% | **50.5%** | 5.03% |
| Stack | 10% | **82.9%** | 9.79% |

Guarantees validate almost exactly on held-out learners; the stack buys ~5 points
of coverage at the 5% risk level.

## What this says

1. **The biggest missing ingredient is item difficulty (IRT).** Word-level
   difficulty intercepts alone add +0.041 AUROC — four times any other single
   addition — and land the MAP model exactly at the official pipeline's level.
   Reframing the model as Bayesian IRT (ability − difficulty) is both the
   narrative upgrade and the predictive one.
2. **The HMM epilogue resolves cleanly.** A *continuous* dynamic-ability signal
   adds real predictive value (+0.006, CI excludes zero) where *discrete* HMM
   stages did not: learner state drifts, it doesn't jump. Dynamic IRT is the
   right future-work framing.
3. **Memory matters (Duolingo's own insight).** Half-life-regression-style
   exposure/lag features add +0.010 — the second-largest win — connecting the
   model to the spaced-repetition literature.
4. **Interpretability costs nothing here.** With the right features, the linear
   hierarchical model *beats* gradient-boosted trees on the identical
   information set (+0.011, significant) and is better calibrated. The
   BART/nonlinearity question is answered: no free lunch from function-class
   flexibility; the structure is additive-logistic to a good approximation.
5. **Stacking is the cheap final polish** (+0.013 over the best single model),
   and conformal gives the abstention layer a guarantee the report can state
   precisely: "at a 5% error budget among accepted tokens, the system safely
   auto-handles ~50% of tokens and routes the rest."

## Caveats before the report

- These are single-dev-set evaluations — the exact setting where the HMM's tiny
  "improvement" fooled us. **Run Richard's forward-validation gate on M3/M5/M6
  before promoting any of them into the final pipeline.** The IRT delta (+0.04)
  is 20× the noise floor and will almost certainly survive; the smaller rungs
  need the gate.
- MAP approximations, not posteriors: for the report's uncertainty story, the
  winning blocks (item difficulties, slopes) should be re-fit in PyMC as
  hierarchical random effects (item effects; LKJ-correlated slopes) if time
  allows — or reported as MAP extensions with the PyMC port as future work.
- The stacker and conformal results use a learner-level cal/test split of dev;
  all other rows use the full dev set.

---

# Cross-language check: the ladder replicated on all three tracks

Same pipeline, fit and evaluated natively per track (official dev sets).

| Model | es_en | fr_en | en_es |
|---|---|---|---|
| M1 pooled | 0.6700 | 0.7056 | 0.6969 |
| M2 + learner | 0.7144 | 0.7317 | 0.7372 |
| M3 + IRT item difficulty | 0.7554 **(+0.041)** | 0.7768 **(+0.045)** | 0.7797 **(+0.043)** |
| M4 + dynamic ability | 0.7611 (+0.006) | 0.7802 (+0.003) | 0.7863 (+0.007) |
| M5 + memory/HLR | 0.7709 (+0.010) | 0.7917 (+0.012) | 0.7947 (+0.008) |
| M6 + learner×format slopes | 0.7800 (+0.009) | **0.8001** (+0.008) | **0.8061** (+0.011) |
| M7 GBM (same info set) | 0.7692 | 0.8018 | 0.7621 |
| Stack (learner-split test half) | 0.7937 | 0.8131 | 0.8119 |
| *Official pipeline (matrix diagonal)* | *0.7554* | *0.7765* | *0.7769* |

Conformal (stack, 5% risk budget): es_en 50.5% coverage (risk 5.03%) · fr_en 50.8%
(5.28%) · en_es 60.0% (5.07%). At 10%: 82.9% / 80.9% / 90.5%.

**Generalization verdict.**
- The ladder's shape is language-universal: the IRT item-difficulty jump is
  +0.041–0.045 on every track (always the largest rung by ~4×), memory adds
  +0.008–0.012, slopes +0.008–0.011, dynamic ability +0.003–0.007.
- Striking anchor: M2+IRT lands within 0.003 of the official pipeline's number on
  *every* track (0.7554/0.7554, 0.7768/0.7765, 0.7797/0.7769) — the ladder first
  reproduces the current pipeline, then everything above it is new headroom
  (+0.025–0.035 with the full GLM, +0.035–0.04 with the stack).
- GLM vs GBM nuance: the interpretable GLM wins clearly on es_en (+0.011) and
  en_es (+0.044, where the GBM is also badly calibrated, ECE 0.037); on fr_en the
  GBM edges it by a statistically marginal +0.002. Verdict: nonlinearity never
  buys a meaningful, consistent gain.
- fr_en's conformal test risk (5.28% vs 5% target) sits slightly over budget —
  within binomial noise of the finite-sample correction on the smallest track,
  worth a footnote, not a worry.
