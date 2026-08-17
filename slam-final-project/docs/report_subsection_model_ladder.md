# Report paragraphs — paste into Final_Report_Learner_Errors.docx

## An Item-Response Reframing and a Ladder of Model Extensions

Our hierarchical logistic model admits a natural psychometric reading: the
learner intercept is an ability parameter and the group-level effects are
difficulty parameters, making the model a hierarchical Rasch (item response
theory) model in all but name. Taking this reading seriously suggests one
structural addition the base model lacks — difficulty parameters at the level
of the individual *item* (word) — and a family of further extensions drawn
from the educational-modeling literature. We evaluated these as an incremental
ladder, each rung fit on the es_en training data and evaluated on the official
development set, with every improvement tested against learner-clustered
bootstrap intervals.

The results are unambiguous about where the remaining signal lives. Word-level
item difficulties are the single largest improvement in the entire project
(+0.041 AUROC, 95% CI [+0.039, +0.043]) — four times the gain of any other
addition — confirming that error prediction in this corpus is, to first order,
an ability-by-difficulty interaction in the classic IRT sense. Three further
rungs each add smaller but reproducible gains: a leakage-safe dynamic-ability
signal summarizing the learner's recent error history (+0.006 [+0.004,
+0.008]); memory features in the spirit of Duolingo's half-life regression —
prior exposures to the word, the learner's error history on it, and the lag
since last exposure (+0.010 [+0.009, +0.011]); and shrunken learner-by-format
random slopes permitting individual differences in modality difficulty (+0.009
[+0.008, +0.011]). The dynamic-ability result gives our Hidden Markov Model
negative finding a satisfying resolution: a *continuous* drift in learner
state carries predictive information where *discrete* acquisition stages did
not, pointing to dynamic IRT — a state-space ability model — as the principled
next step.

Two further comparisons discipline the ladder. First, a gradient-boosted tree
ensemble trained on the identical information set — a stand-in for flexible
nonparametric alternatives such as BART — performs significantly *worse* than
the final linear model (0.769 vs. 0.780, Δ = +0.011 [+0.007, +0.013]) with
worse calibration, indicating that the additive-logistic structure of the
hierarchical model is not a limiting approximation on this task and that
interpretability carries no predictive cost. Second, a logistic stacking
ensemble of all seven models, fit on a learner-level split, adds a final
+0.013 [+0.010, +0.016] over the best single model (0.794 on held-out
learners), quantifying how much complementary signal remains distributed
across model variants.

We emphasize a methodological caution learned earlier in this project: these
are development-set evaluations, the same regime in which the HMM hybrid's
apparent improvement failed forward validation. The item-difficulty gain is
roughly twenty times the sampling noise floor and is robust to any plausible
correction; the smaller rungs are reported here as significant on the
development set and are subject to the same forward-validation gate before
inclusion in the final pipeline. All extensions are MAP estimates of
hierarchical priors and port directly to the fully Bayesian pipeline (item
difficulties as an item-level random effect; slopes as LKJ-correlated random
effects).

## Conformal Guarantees for the Routing Decision (add to abstention section)

Our Bayesian abstention layer selects a routing threshold by expected utility
under the posterior; its guarantees are only as good as the posterior's
calibration. As a distribution-free complement, we add split-conformal
selective classification: a threshold chosen on a learner-level calibration
split with a finite-sample correction, guaranteeing that the error rate among
auto-handled tokens stays below a target budget without any modeling
assumptions. The guarantees validate almost exactly on held-out learners: at a
5% risk budget the system safely auto-handles 50.5% of tokens (empirical risk
5.03%), and at 10% it auto-handles 82.9% (9.79%). The practical statement for
an adaptive tutor is precise: half of all tokens can be handled automatically
while provably keeping the error rate among them at one in twenty, with the
remainder routed to review — the "knows the limits of its own knowledge"
property, now with a guarantee.

## Future work (one sentence each, for the Discussion)

Dynamic item response theory — a state-space model of continuously drifting
ability — is the natural successor to both our HMM analysis and the
dynamic-ability feature; half-life regression proper would model forgetting
explicitly from the exposure lags our memory features summarize; and
LKJ-correlated random slopes in the full PyMC pipeline would give posterior
uncertainty on the individual-differences structure that our MAP slopes
capture pointwise.
