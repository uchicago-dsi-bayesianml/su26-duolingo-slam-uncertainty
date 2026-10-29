<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Uncertainty-Aware Modeling of Second-Language Acquisition

> **Example project from Summer 2026.** This is a copy of [ChicagoShane/slam-final-project](https://github.com/ChicagoShane/slam-final-project), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project) and three short checks in `staff/`. The Duolingo SLAM data are public on Harvard Dataverse under CC BY-NC 4.0 and are not in this repository, apart from the 600 dev tokens in the demo's data file; [data/README.md](data/README.md) explains how to get them.

A language app has to decide what to show a learner next. It does better if it knows where the learner is about to make a mistake, and better still if it knows when its own guess is unreliable and can hand the case to a person. The team took the 2018 Duolingo shared task on second-language acquisition modeling (SLAM), in which every word of a learner's answer is labelled right or wrong, and built a hierarchical Bayesian logistic regression in PyMC for the French course: about 927,000 training tokens, learner and word effects with learned pooling, random slopes for forgetting, learning and exercise format, and a low-rank learner-by-word interaction, fit by minibatch variational inference. Alongside it they built a scikit-learn line on all three courses: a ladder of penalized logistic models that adds one block of structure at a time, a study of why a model fit on one course does worse on another, and an abstention rule that decides which predictions a tutor may act on alone. A browser demo shows dev tokens with the posterior draws behind each prediction.

## Summary

**Data.** The SLAM 2018 corpus: learners' answers to Duolingo exercises over their first 30 days in three courses, English speakers learning Spanish (es_en) and French (fr_en), and Spanish speakers learning English (en_es). Every token carries an error label, learner and exercise metadata, and linguistic tags. The official dev set holds each learner's later exercises, so the same learners appear in training and dev. The PyMC notebooks use the French course: 926,646 training tokens with an error rate of 16.2 percent and 137,564 dev tokens (`analysis/slam_bayesian_final_refit_commented.ipynb`, cell 9).

**Features.** History features use only earlier exercises: the learner's smoothed past error rate overall and on the word, exposures and time since the last, decayed error and success counts, and word and construction difficulty from earlier exercises of all learners. With splines and ability-by-difficulty terms, the final model has 35 numeric and 13 categorical predictors (cell 37).

**Model.** A Bernoulli logit model with pooled learner, word and morphology effects, learner recency slopes, word learning slopes, learner-by-format effects and a rank-2 learner-by-word factor model, its structure chosen from all 64 combinations of six optional extensions. Mean-field ADVI in PyMC 5.28.5 on minibatches of 8,192 tokens; 1,000 draws give every dev token a posterior mean and standard deviation.

**The scikit-learn line.** L2-penalized logistic regressions, "MAP analogues" in the write-up, on all three courses: pooled, then adding learner intercepts, word intercepts, history, memory and learner-by-format effects, plus gradient boosting and a stack. Further scripts study the AUROC drop under transfer between courses and choose an abstention threshold for a 5 or 10 percent error budget.

**Demo.** `demo.html` shows 300 French and 300 Spanish dev tokens from `demo_data.json`, each with 120 posterior draws from the French-course model, aleatoric and epistemic bars, and a verdict set by an abstention slider.

The staff section below sorts the reported numbers by where they came from, and looks at model selection on the dev set, labels leaking into the ladder's features, what the abstention rule guarantees, and the wording of the transfer results.

## Team and contributions

Team "Optimus Primes", Summer 2026.

- **Richard Pollitt** (no GitHub account in the history): the PyMC hierarchical model, that is, the notebooks' parser, the leakage-safe feature registry, the L2 baseline, the hierarchical model and its extensions, the 64-configuration grid, the forward cross-validation and the final refit, in the two notebooks and four HTML renders in `analysis/`. All of them were run under `C:\Users\Richard\...`, where their configuration cells read the data and write their output (cell 5 of the final refit, for example), and `analysis/forward_gate.py` calls its out-of-time validation "Richard's protocol". The team's task tracker, in the first commit, assigned Richard the data engineering and the baselines.
- **Shane Dunkle** ([@ChicagoShane](https://github.com/ChicagoShane)): every commit; the README that is now REPORT.md; the scikit-learn scripts in `analysis/*.py`, covering the ladder, stack and abstention rule, the transfer study and the forward gate; the reports in `docs/`; the deck and the generators in `tools/`; the demo page and its data file, although the code that made the file is not here; and the project tracker, which assigned Shane the Bayesian modeling and the interpretation.

The repository ran from July 12 to August 17, 2026: the tracker first, the analysis, documents and deck in two commits on August 16, and the demo, notebooks and renders on August 17. All ten commits are from Shane Dunkle's account, so the committing account is not the only author; the attributions follow the paths inside the notebooks, the scripts' docstrings and the [commit history](https://github.com/uchicago-dsi-bayesianml/su26-duolingo-slam-uncertainty/commits/main).

## Getting started

The data are not in the repository; [data/README.md](data/README.md) explains the download and where the code expects the files. The notebooks keep their outputs and can be read without it.

`requirements.txt` covers the scripts. The notebooks also need PyMC (the recorded runs used 5.28.5), ArviZ and joblib; the deck generators need Node with `pptxgenjs` and `docx`.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pymc arviz joblib jupyter
```

**The PyMC pipeline**, French course only. Set `DATA_DIR` in the first configuration cell of each notebook.

1. `analysis/slam_modular_bayesian.ipynb`: parsing, features, leakage checks, the L2 baseline and a first hierarchical fit. It writes the feature tables that the grid reads.
2. The extension grid, 64 fits of 15,000 ADVI steps. Only its render, `analysis/slam_bayesian_extension_grid.html`, is committed.
3. `analysis/slam_bayesian_final_refit_commented.ipynb`: the locked configuration fit on all training tokens and scored on dev, with the posterior and predictions saved. The fit took 36.6 minutes (cell 39).

**The scikit-learn pipeline**, from the repository root with the data under `slam/<track>/`: `analysis/parse_slam.py` once per file; `fit_models.py`, `bootstrap_analysis.py`, `fit_variance.py`, `analysis3a_recal.py`, `analysis3b_covshift.py` and `analysis4_joint.py` for the transfer study; `ladder_track.py <track>` per course; `build_seq_features.py` then `forward_gate.py` for the gate; and `tools/make_figures.py`, which writes the figures into the current folder.

**The demo.** The authors publish it at [chicagoshane.github.io/slam-final-project/demo.html](https://chicagoshane.github.io/slam-final-project/demo.html); this copy has no Pages site. `demo.html` also opens from a local checkout. If the browser will not load `demo_data.json` from a local file, the page offers a link to pick the file, or run `python -m http.server` in the folder and open `localhost:8000/demo.html`.

**Staff checks.** None needs the data; Results describes what they find, and the top of each file says what it cannot cover.

```sh
python staff/check_provenance.py               # standard library only, under a second
python staff/check_demo_uncertainty.py         # NumPy, under a second
python staff/check_conformal_rule.py --small   # NumPy, about two minutes
```

**What does not run as committed.** `analysis/parse_slam.py` writes no `morph` column, while `fit_models.py` (line 22), `analysis3a_recal.py` (50), `analysis3b_covshift.py` (26) and `analysis4_joint.py` (61) read `df["morph"]`, and `ladder_track.py`, `fit_variance.py` and `forward_gate.py` import the same code, so every script that fits a model stops with `KeyError: 'morph'`; the authors' Parquet files must have come from another version of the parser. `stack_conformal.py` and `tools/make_figures.py` read `slam/ladder_preds.npz`, which no script writes (`ladder_track.py` writes `slam/ladder_preds_<track>.npz`). `analysis/clustered_bootstrap_cell.py` is a cell for the Colab notebooks described under Results. The notebooks use Windows paths, and two of the four runs exist only as HTML.

## Data

`.gitignore` keeps the data out: the `slam/` folder, the raw `*.slam.20190204.*` files, Parquet tables, NumPy archives and the notebooks' `processed/` folder.

- **Source:** [Data for the 2018 Duolingo Shared Task on Second Language Acquisition Modeling](https://doi.org/10.7910/DVN/8SWHNO) (Settles et al., 2018), Harvard Dataverse: one archive per course, 16 to 47 MB compressed, and the shared task's starter code.
- **Terms:** CC BY-NC 4.0, non-commercial use with attribution. Dataverse asks for a guestbook response before it releases the files.
- **Splits:** the official train and dev files; nothing uses the test set.
- **Committed:** `demo_data.json`, 600 dev tokens with their text, tags, exercise metadata, label, prediction and posterior draws, without learner IDs. The authors' data note says no derived data are committed; the staff section of that note corrects this.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' README. |
| `analysis/slam_modular_bayesian.ipynb` and `.html` | Richard Pollitt's first full pipeline on fr_en, with a hierarchical fit; the HTML renders the same run. |
| `analysis/slam_modular_bayesian - pruned.html` | A later run of that notebook with the five selected extensions, SHAP values, posterior summaries and the forward cross-validation. No `.ipynb`. |
| `analysis/slam_bayesian_extension_grid.html` | Render of the 64-configuration grid, each fit scored on dev. No `.ipynb`. |
| `analysis/slam_bayesian_final_refit_commented.ipynb` and `.html` | The final refit with metrics and diagnostic plots; the HTML renders the same run. |
| `analysis/parse_slam.py`, `fit_models.py` | Shane Dunkle's parser and the shared design matrices of the scikit-learn line. |
| `analysis/ladder_track.py`, `build_seq_features.py`, `forward_gate.py`, `stack_conformal.py` | The ladder per course with history features, stack and abstention rule; the out-of-time gate; an earlier es_en-only stack. |
| `analysis/bootstrap_analysis.py`, `fit_variance.py`, `analysis3*.py`, `analysis4_joint.py`, `clustered_bootstrap_cell.py` | The transfer study. |
| `docs/` | Four reports on the ladder and the transfer study; the deck and presenter scripts in `presentation/`. |
| `figures/`, `tools/` | The three figures; generators for figures, deck and script. |
| `demo.html`, `demo_data.json`, `staff/` | The demo and its 600 tokens; course staff's three checks. |

## Method

**Features in the notebooks.** Each feature family is a registered function switched on in the configuration cell, and the build asserts after every family that no row was added, lost or reordered. History is aggregated per learner, word and exercise and shifted by one exercise, so no token sees its own exercise's labels; global difficulty uses file order as its clock. Dev labels are blanked before any feature is built (final refit, cell 30) and joined back only after the design matrices are fixed (cell 37). Unseen categories get an effect fixed at zero.

**Model.** For a token of learner *u* on word *w* in format *f*,

> logit P(error) = α + **x**ᵀβ + Σ_c γ_c[level of c] + ρ[u] · recency + λ[w] · exposure + δ[u, f] + s · ⟨**a**[u], **b**[w]⟩ / √2,

with α ~ Normal(0, 1.5) and β ~ Normal(0, 0.35). Each categorical effect is a vector of Normal(0, 1) offsets, centred and multiplied by a scale: a learned HalfNormal(1) scale for learners and words, a learned HalfNormal(0.35) for each retained morphology feature, and a fixed 0.35 otherwise. The slopes ρ and λ and the format effects δ are centred offsets times learned HalfNormal(0.35) scales; the rank-2 factors **a** and **b** are Normal(0, 1) with a learned scale *s*. A part-of-speech hierarchy over words was tested and left out.

**Inference.** `pm.fit` with `pm.Minibatch` and `total_size` set to the training size, so the minibatch objective estimates the full-data ELBO; Adam at 0.01 for 30,000 steps. The convergence callback never stopped the run early ("Finished [100%]", cell 39), and the ELBO curve in cell 41 flattens within a few thousand steps.

**Model selection.** The grid ranked all 64 subsets of the six extensions by AUROC on the official dev labels (grid render, cells 4 and 16 to 18). The final refit's prior scales and smoothing came from an "overnight within-user temporal search" that is switched off in the committed run. The forward cross-validation cuts the training file's exercises into six consecutive blocks with `TimeSeriesSplit`, validates on each of the last five after training on the earlier ones, and rebuilds every feature with the validation labels masked.

**The scikit-learn ladder.** L2 logistic regressions with C = 1 on sparse blocks: M1 exercise metadata, tags, log days and log time; M2 adds one-hot learners; M3 words seen 20 or more times; M4 three history features; M5 five memory features on the word; M6 a learner-by-format block scaled by 0.5, a tighter prior. M7 is gradient boosting, and the stack a logistic regression on the seven models' logits, fit on half of the dev learners. On that half the abstention rule scans 96 quantiles of the stack's error probability and keeps the largest threshold at which (errors + 1) / (tokens + 1) below it is within budget. For transfer, models fit on each course predict the es_en dev set without learner effects for unknown learners, followed by a clustered bootstrap, refits, intercept-only recalibration, importance weighting and a joint three-course model.

### What is Bayesian here

Richard Pollitt's model is a real hierarchical Bayesian model at a scale few course projects attempt. Learners, words and morphology features get effects whose spread is learned, so sparse learners and rare words are pulled toward the population, which is partial pooling as in the course's hierarchical-model sessions; the random slopes and the low-rank interaction carry the idea to forgetting, learning and learner-word affinity. The posterior is approximated by stochastic variational inference with a correctly scaled minibatch ELBO, and predictions average over draws rather than plugging in a point estimate. The demo splits each token's uncertainty into aleatoric and epistemic parts from the draws.

What it stops short of: the approximation is mean-field, which understates posterior spread, and nothing checks it against MCMC. The tracker in the first commit planned NUTS on a subsample with R-hat diagnostics, but no sampler is called anywhere, so the write-up's "fit with NUTS/ADVI" and the deck's "MCMC (NUTS)" are not borne out. No LOO or WAIC is computed in committed code, and the posterior standard deviation enters no reported routing decision.

The scikit-learn line is not Bayesian. An L2 logistic regression is the MAP estimate under a Gaussian prior of fixed variance, so "MAP analogue" is fair, but no variance is learned, so it has fixed shrinkage rather than partial pooling and no posterior uncertainty. The abstention rule is a frequentist threshold, and the HMM in the write-up has no code here.

## Results

The numbers come from three places. `staff/check_provenance.py` prints group (a) from the committed outputs and finds none of the headline numbers of (b) and (c) in any of them.

### (a) The PyMC notebooks: committed with outputs

French course, 137,564 dev tokens.

| Model | Dev AUROC | Source |
|---|---:|---|
| L2 logistic regression, same features | 0.8017 | `slam_modular_bayesian.ipynb`, cell 30 |
| Hierarchical, learner recency slopes only | 0.8020 | same, cell 37 |
| L2 logistic regression, later run | 0.8013 | `slam_modular_bayesian - pruned.html`, cell 32 |
| Hierarchical, five selected extensions, prior scales 0.5 | 0.8141 | same, cell 43 |
| Extension grid, 64 configurations at 15,000 steps | 0.8006 to 0.8131 | `slam_bayesian_extension_grid.html`, cells 16, 18 |
| Final refit, tuned scales, splines and interactions | 0.8139 | `slam_bayesian_final_refit_commented.ipynb`, cell 41 |

The final refit also reports a PR-AUC of 0.547, a Brier score of 0.111, a log loss of 0.360, an ECE of 0.006 over 15 bins, 90.5 percent accuracy on its most confident 80 percent of tokens, and a mean posterior standard deviation of the error probability of 0.030 (cell 41). In the grid, learner-by-format effects alone take the AUROC from 0.801 to 0.811 (cells 16 and 18), most of the gain; the final refit's extra tuning leaves it where the later run had it.

**Five-fold forward cross-validation** (later run, cell 49): 0.7974, 0.7950, 0.7892, 0.7986 and 0.7979, a mean of 0.7956 with a standard deviation of 0.0038.

### (b) The scikit-learn scripts: code here, no outputs

The authors' numbers from `docs/model_ladder_comparison.md`; the scripts do not run as committed and saved no output.

| Dev AUROC as reported | es_en | fr_en | en_es |
|---|---:|---:|---:|
| M1 pooled | 0.6700 | 0.7056 | 0.6969 |
| M2 + learners | 0.7144 | 0.7317 | 0.7372 |
| M3 + words | 0.7554 | 0.7768 | 0.7797 |
| M4 + history | 0.7611 | 0.7802 | 0.7863 |
| M5 + memory | 0.7709 | 0.7917 | 0.7947 |
| M6 + learner-by-format | 0.7800 | 0.8001 | 0.8061 |
| M7 gradient boosting | 0.7692 | 0.8018 | 0.7621 |
| Stack, test half of the learners | 0.7937 | 0.8131 | 0.8119 |

Reported abstention for the stack at a 5 percent budget: 50.5, 50.8 and 60.0 percent of test tokens auto-handled on es_en, fr_en and en_es, at error rates of 5.03, 5.28 and 5.07 percent. The transfer diagnostics the lessons quote, from `docs/transfer_gap_deep_dive.md`, come from the same scripts.

### (c) The Colab pipeline: not in this repository

The write-up's headline table and the deck rest mostly on notebooks that the documents call "the official pipeline" and `analysis/clustered_bootstrap_cell.py` calls the Colab "CrossTrack notebook". None of these numbers can be reproduced from this repository:

- native AUROC of 0.755, 0.777 and 0.777 on es_en, fr_en and en_es;
- the 3 × 3 transfer matrix in `figures/fig1_transfer_matrix.png`, whose nine values are typed into `tools/make_figures.py` at line 19, and the 21 values of the ladder plot, typed in at line 99; only the risk-coverage figure is computed from saved predictions;
- a transfer AUROC of 0.732 with a 95 percent interval of [0.727, 0.737];
- an ELPD gain of 2,158 ± 77 by PSIS-LOO, which the deck's notes attribute to a `run_summary.json` that is not committed;
- ECE of 0.020 pooled against 0.014 hierarchical; the only ECE in a committed output is the final refit's 0.006;
- the HMM result, a log-loss increase of 0.00055 with an interval of [0.00018, 0.00092];
- epistemic uncertainty rising 4.6 times under transfer.

### Staff checks

**Provenance.** Every committed AUROC is on the French course. Only the deck's "0.814" matches a committed output, and that output is the hierarchical model, not the baseline the deck names. There is `pm.fit` in all six notebook sources and no sampler, LOO, WAIC, HMM or conformal library in any code; "NUTS" occurs only in slide text and in a comment that "full-data NUTS is impractical at this scale".

**The demo's uncertainty.** `staff/check_demo_uncertainty.py` confirms the standard decomposition: p is the mean of the draws, "aleatoric" their expected entropy, "epistemic" the mutual information. The epistemic part is a median of 0.4 percent of the total uncertainty on the French tokens and 1.7 percent on the Spanish ones. On the 60 French and 60 Spanish tokens labelled "random", the only subset not selected for being extreme, mean epistemic uncertainty is 0.0016 and 0.0076 nats, a ratio of 4.79 with a bootstrap 95 percent interval of [3.74, 6.15], consistent with the deck's 4.6. The demo's routing reads only p, deferring when |p − 0.5| is below the slider; since epistemic uncertainty peaks near p = 0.5 (correlations of −0.59 and −0.90 with |p − 0.5|), it still defers on 88 and 100 percent of the fifth of tokens with the most epistemic uncertainty.

**The abstention rule.** `staff/check_conformal_rule.py` applies the authors' rule to 200 simulated dev sets, calibrated by construction but with part of each learner's propensity hidden from the model, split by learner as the authors did, and computes each chosen threshold's true error rate among auto-handled tokens from a large independent sample.

| es_en-sized, 200 replicates | Budget | Auto-handled | True risk over budget |
|---|---:|---:|---:|
| Authors' (errors + 1) / (n + 1) | 5% | 57.3% | 34.5% of replicates |
| Wilson bound, tokens independent | 5% | 56.3% | 6.5% |
| Normal bound, variance over learners | 5% | 55.8% | 2.0% |
| Authors' (errors + 1) / (n + 1) | 10% | 84.6% | 33.0% |
| Wilson bound, tokens independent | 10% | 84.1% | 12.5% |
| Normal bound, variance over learners | 10% | 83.4% | 2.5% |

At the fr_en size the authors' rule leaves the true risk over budget in 37.5 and 40.5 percent of replicates, and the learner-level bound in 4.0 percent for both budgets. The absolute numbers depend on the simulation's settings, which the script prints; the pattern does not.

## What to learn from this project

Course staff added this section. The PyMC model is careful, substantial work; the points below are about how it and the rest of the project are reported.

- **Say which pipeline every number came from.** The write-up's headline table mixes notebooks committed with outputs, scripts committed without any, and Colab notebooks that are not here; most headline numbers come from the last group, and two of the three figures are drawn from typed-in numbers. The one fully reproducible result, Richard Pollitt's model at 0.8139 on French dev, is absent from the table, and its notebooks are missing from the repository map. Every row of a results table should name the file and cell, or the log, behind it.
- **Keep the dev set out of model selection, or say that you did not.** The grid ranks its 64 configurations by dev AUROC and the final refit takes the winner, yet that notebook describes its configuration as the result of "training-only screening" and says "no hyperparameter is changed after viewing these results". The top configurations sit within 0.001 of each other, so the optimism is probably small, but 0.8139 is no longer a clean held-out estimate. The forward cross-validation, 0.7956 ± 0.0038, is the honest number, with a caveat: its folds follow file order rather than each learner's timeline, so they are not the dev set's task of predicting the same learners' later exercises.
- **Beware features that have seen the answer.** In the scikit-learn ladder, dev rows get their labels from the key file, and `ladder_track.py` and `build_seq_features.py` concatenate train and dev and update each learner's history after every token. A token's "past error rate" therefore includes the labels of earlier words in the same answer, and every dev token benefits from earlier dev labels. Richard Pollitt's notebooks avoid both by shifting at the level of the exercise and deleting dev labels first. The leak was not measured, but rungs M4 to M6 and the gate result for the full ladder are not comparable with the notebooks.
- **A plug-in threshold is not a guarantee.** The write-up calls the abstention rule split-conformal with "finite-sample risk control", and the deck's notes say the tutor handles the easy half "provably safely". The rule keeps the largest threshold whose smoothed empirical error rate is within budget, which keeps the risk near the budget but not below it with any stated probability: the staff simulation finds the true risk over budget in about a third of replicates. The stack is also fit on the calibration half that sets the threshold, which split-conformal arguments do not allow. The reported fr_en test risk of 5.28 percent on a 5 percent budget, put down to binomial noise, is what this rule produces; in the fr_en-sized simulation its test-half error rate reached 5.62 percent. A bound computed over learners, the unit of the split, kept the true risk within budget for one to two points of coverage.
- **Label comparisons correctly.** The deck lists a "tuned non-Bayesian logistic baseline: AUROC 0.814"; the only 0.814 in the repository is the final hierarchical model, and the L2 baseline is 0.8017. The write-up calls word-level difficulty "the dominant missing signal", but Richard Pollitt's model had a pooled word effect from the start; it is missing only from the ladder's M2. In the notebooks the hierarchical prior alone adds almost nothing over L2, 0.8020 against 0.8017; the gain to 0.81 comes from the learner-by-format effects and other extensions.
- **Word transfer findings to match their numbers.** The write-up says refitting group intercepts on about 10 percent of target learners "recovers most native performance". In the authors' replication it takes AUROC from 0.6220 to 0.6624 against a native 0.7145, 44 percent of the gap; it recovers 84 percent of the gap to the native model without learner intercepts, which is what the deep-dive document says. About half the gap is that a model from another course has no intercepts for the es_en learners, and reweighting with an effective sample size of 18 percent is weak evidence against covariate shift. The "difficulty geometry" is constrained by construction: if three deviation vectors sum to roughly zero, the cosine between two of them is (|c|² − |a|² − |b|²) / (2|a||b|), so the pair that excludes the longest vector is always the closest. "The mirror pair clusters" and "fr_en is the outlier with the largest deviation norm" are one observation, and neither has an interval.
- **Ship the code for a negative result, and tell it one way.** The HMM's failure under forward validation is presented as a strength, and would be one, but no HMM code is here and the accounts disagree: the write-up and deck describe three states, the presenter script says the two-state model was the strongest; the deck says "in all four forward folds", `forward_gate.py` has three folds and no HMM, and the notebooks' cross-validation has five.
- **Make the committed code run.** See "What does not run as committed"; in addition, `requirements.txt` leaves out PyMC and ArviZ, and the deck credits NumPyro and hmmlearn, which nothing uses.

What the project does well: a hierarchical Bayesian model fit by minibatch variational inference to almost a million tokens, with learned pooling scales, random slopes and a low-rank interaction; features that are leakage-safe by design, with assertions after every step; a modular registry that makes ablations a one-line change, and a full factorial ablation of the extensions; a forward cross-validation that rebuilds every feature in every fold; good calibration, checked with a reliability diagram; per-token posterior uncertainty in a demo anyone can open; transfer diagnostics that separate evaluation noise from refit noise; a readiness to report negative results; and a problem a new student can grasp in a minute.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Summer 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Added rules to `.gitignore` for the raw SLAM files, the notebooks' `processed/` folder and local environments. No clutter was found. All four HTML renders were kept: none is a byte-identical copy of another, two render the committed notebooks, and two are the only record of their runs.
- Added a staff section to `data/README.md` on the Dataverse terms and download, where the code expects the files, and `demo_data.json` as a derived artifact. The authors' text above it is unchanged.
- Added `staff/check_provenance.py`, `staff/check_demo_uncertainty.py` and `staff/check_conformal_rule.py`, described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors' `LICENSE` is the MIT License, copyright 2026 Shane Dunkle and Richard Pollitt, followed by a note that it covers the code and documentation only and not the SLAM data. Because of that appended note GitHub does not recognize the file as MIT and shows the license as "Other"; course staff left it unchanged. The SLAM data are CC BY-NC 4.0, so `demo_data.json` and anything else derived from them is for non-commercial use, with the citation in [data/README.md](data/README.md).
