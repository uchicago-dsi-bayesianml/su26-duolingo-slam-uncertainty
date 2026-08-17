# Cross-track transfer gap: deep-dive results (Analyses 1–4)

**Question.** Is the fr_en→es_en transfer drop (0.732 vs 0.755 native in our official
matrix) sampling variation, and if not, where does it live?

**Setup.** Independent replication in a clean environment: MAP analogue of the
hierarchical logistic model (shared block: format, session, client, country, POS,
dependency, morphology, log-days, log-time; learner intercepts), fit per track on the
SLAM training sets, evaluated on the real es_en dev (288,864 tokens, 2,640 learners).
Absolute AUROCs run below the PyMC pipeline (leaner features): native 0.7145,
fr_en transfer 0.6220, en_es transfer 0.6107. Noise estimates and mechanisms are the
transferable results; run the companion bootstrap cell on the saved posterior draws
for headline CIs.

## 1. Learner-clustered bootstrap (B=1,000) — the gap is real, the *ranking* is not
- AUROC bootstrap SE on es_en dev: 0.0025–0.0032; paired-delta 95% CIs ≈ ±0.003.
- Applied to the official numbers: native−transfer = 0.023 ≈ 5× noise → **real**.
- fr_en vs en_es transfer difference (0.003) is **within noise** — "fr_en is the worst
  transfer" is not a stable ranking (it flipped in replication: fr_en − en_es =
  +0.011 [+0.008, +0.015]).

## 2. Fit-side variance (learner-bootstrap refits) — not an unlucky fit
- 10 fr_en refits: transfer AUROC SD 0.0022 (range 0.0066). 5 es_en refits: SD 0.0005.
- Combined eval+fit noise ≈ 0.004 ≪ gap.

## 3a. Recalibration-only transfer — the gap is in group-level effect sizes
Freeze the fr_en shared coefficients; refit only target-side intercepts on es_en train
(French linear predictor carried over as one feature):

| tier | target-side refit | AUROC |
|---|---|---|
| T0 | none (raw transfer) | 0.6220 |
| T1 | group intercepts (format/POS/dep/morph/country) | **0.6710** |
| T1s | same, only 10% of learners | 0.6624 |
| T2 | T1 + learner intercepts | **0.7139** |
| ref | native es_en model | 0.7145 |
| ref | native, learner ints zeroed | 0.6699 |

T1 fully recovers the native global-only level; T2 fully recovers native. **The frozen
French structure carries all recoverable signal** — the deficit is (a) group-level
effect sizes and (b) learner effects, in roughly equal parts; and most of (a) is
recoverable from ~10% of target learners.

## 3b. Covariate shift — ruled out
- Marginals shift moderately (TV: format 0.25, country 0.26, session 0.15; days
  distributions differ), and a domain classifier separates tracks easily (AUROC 0.91).
- But importance-weighting fr_en training toward the es_en feature distribution
  (clipped odds weights, ESS 18%) makes transfer **worse**: 0.6003 vs 0.6220.
- ⇒ The gap is **concept shift** — P(error | features) differs across course
  languages — not input-distribution mismatch. Consistent with 3a.

## 4. Joint multi-track model — the latent-space readout
All three tracks fit jointly: global coefficients + shrunk per-track deviation blocks
(partial pooling) + learner intercepts. Pairwise cosine similarity of deviation
vectors (chance baseline ≈ −0.5 under the near-sum-to-zero constraint):

| pair | overall | linguistic (POS+dep+morph) | session | numeric |
|---|---|---|---|---|
| es_en ~ en_es (mirror pair) | **−0.18** | **−0.15** | +0.83 | +0.53 |
| fr_en ~ es_en (same L1 population) | −0.51 | −0.50 | −0.98 | −0.80 |
| fr_en ~ en_es | −0.65 | −0.70 | −0.88 | −0.90 |

- **The mirror pair (es_en, en_es) is the closest pair** — well above the chance
  baseline — while fr_en is the outlier (largest deviation norm, 3.09). Difficulty
  structure follows the *language contrast* (Spanish↔English), not the learner
  population: fr_en/es_en share L1-English learners (only 2 actual shared users) yet
  sit at chance-level similarity.
- Predictive check (es_en dev, no learner ints): correct deviation block 0.670,
  global-only 0.652, wrong blocks 0.633 (fr) / 0.618 (en_es) — wrong-track adaptation
  is worse than no adaptation.

## Bottom line
The transfer penalty is a reproducible **concept shift in group-level effect sizes**,
not noise and not covariate shift; it is largely erasable with a cheap group-intercept
recalibration on a small slice of target data. The cross-track similarity structure
supports the "languages as points in a latent space" framing: mirror language pairs
share difficulty geometry; same-population tracks do not. For the report: present the
transfer penalty as real, decline to rank transfer sources (within noise), and offer
recalibration-transfer as the practical adaptation recipe.

*Methods caveat: MAP approximations with a leaner feature set than the official PyMC
pipeline; point estimates for the joint-model cosines (no posterior). Days
distributions partially conflate train/dev windows in the TV comparison.*
