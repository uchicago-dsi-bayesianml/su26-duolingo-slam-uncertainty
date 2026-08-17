# Report subsection draft — paste into Final_Report_Learner_Errors.docx

## Cross-Track Transfer: Is the Gap Real, and Where Does It Live?

Applying each track's fitted hierarchical model to the other tracks' development
sets produced an asymmetric transfer matrix: the model fit on fr_en achieved an
AUROC of 0.732 on es_en development data, against 0.755 for the natively fit
model, with the en_es-fit model performing comparably (0.729). Because both
fr_en and es_en draw from the same L1-English learner population, this penalty
invited two competing explanations: a genuine cross-course difference in the
error-generating process, or sampling variation. We investigated with four
companion analyses (run as MAP approximations of the hierarchical model on the
full public data; absolute AUROCs differ from the main pipeline, and we treat
noise magnitudes and mechanisms, rather than point values, as the transferable
results).

First, a learner-clustered bootstrap (B = 1,000, resampling the 2,640
development-set learners with replacement) bounds evaluation-side sampling
noise at a standard error of roughly 0.003 on AUROC, with paired-difference 95%
intervals of approximately ±0.003. The native-versus-transfer gap of 0.023 is
therefore about five times larger than sampling noise and clearly reproducible.
The *ordering among transfer sources*, by contrast, is not: the observed
fr_en-versus-en_es difference (0.003) sits inside the noise band, and indeed
reversed sign in our replication. We accordingly report the transfer penalty as
real but decline to rank transfer sources. Second, refitting the source model
on ten learner-bootstrapped training populations bounds fit-side variability at
a standard deviation of 0.002 — the gap is not an artifact of one unlucky
posterior.

Third, we localized the gap. Freezing every shared coefficient of the fr_en
model and refitting only group-level intercepts (exercise format,
part-of-speech, morphology, dependency, and country) on es_en training data
recovered performance statistically indistinguishable from the native model
evaluated without learner effects; adding learner intercepts recovered native
performance entirely, and a low-resource variant using only 10% of target
learners recovered most of the group-level adjustment. Meanwhile, a
covariate-shift account fails: although the two tracks are easily separable in
feature space (domain-classifier AUROC 0.91), importance-weighting the fr_en
training distribution toward es_en made transfer *worse*. The transfer penalty
is thus a concept shift concentrated in group-level effect sizes — the same
features carry different difficulty in different courses — and is largely
erasable with a cheap group-intercept recalibration on a small slice of target
data, a practically useful recipe for deploying an error model on a new course.

Finally, a joint model over all three tracks — global coefficients plus shrunk
per-track deviation blocks, a partial-pooling structure across courses — offers
a quantitative version of the "languages as points in a latent space" framing.
Against a chance baseline of approximately −0.5 (implied by the near
sum-to-zero constraint on three deviation blocks), the mirror pair es_en and
en_es are by far the closest in deviation space (cosine −0.18 overall, −0.15 on
linguistic features, with session and pacing effects strongly positively
aligned), while fr_en is the outlier with the largest deviation norm.
Difficulty geometry, in other words, follows the language contrast being
learned rather than the learner population: the two tracks that share the
Spanish–English contrast align, while the two tracks that share L1-English
learners do not exceed chance similarity. This connects directly to our
deferral analysis: epistemic uncertainty rises sharply under cross-track
application (mean epistemic variance 4.6× the in-domain level), and the model's
own uncertainty correctly identifies the regime in which its group-level
difficulty map — not its structure — should be re-estimated before trusting its
predictions.

---

### Presentation bullet (one slide)

**Transfer gap: real, localized, and fixable.** Clustered bootstrap: penalty =
5× sampling noise (source *ranking* within noise). Not covariate shift —
reweighting hurts. Group-intercept recalibration on ~10% of target learners
recovers most native performance. Latent-space readout: mirror language pair
(es_en ~ en_es) shares difficulty geometry; same-L1 pair does not.
