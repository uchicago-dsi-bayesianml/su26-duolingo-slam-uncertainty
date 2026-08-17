const { Document, Packer, Paragraph, TextRun, HeadingLevel } = require("docx");

function slide(heading, time, paras) {
  const out = [
    new Paragraph({ heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: heading, bold: true })], spacing: { before: 300, after: 60 } }),
    new Paragraph({ children: [new TextRun({ text: "Target time: " + time, italics: true })],
      spacing: { after: 140 } }),
  ];
  paras.forEach(p => out.push(new Paragraph({ children: [new TextRun(p)], spacing: { after: 140 } })));
  return out;
}

const children = [];

children.push(...slide("Slide 6 — Cross-course transfer", "1 minute, 20 seconds", [
  "Having established the predictive model, we asked whether what it learns in one course transfers to another. The heatmap shows the full transfer matrix: each model fit on one course and evaluated on every course's development set. The diagonal is native performance; every off-diagonal entry is lower.",
  "The transfer penalty is real, not sampling variation. We ran a learner-clustered bootstrap directly on the fitted model's posterior predictions: transfer AUROC into the Spanish course is 0.732, with a 95 percent interval from 0.727 to 0.737 — the native value of 0.755 lies far outside that interval, roughly five times the sampling noise. Refitting the source model on resampled training populations bounds fit-side variability at about one-fifth of the gap, so it is not an artifact of one unlucky fit either.",
  "What is within noise is the ranking of transfer sources. Which course transfers worst differs by only 0.003 — inside the interval — so we report the penalty but decline to rank the sources.",
  "We then diagnosed the mechanism. It is not covariate shift: the courses are easily distinguishable in feature space, but reweighting the source data toward the target's feature distribution actually made transfer worse. It is concept shift — the same features carry different difficulty in different courses. Consistent with that, freezing every shared coefficient and refitting only group-level intercepts on target data recovered essentially all recoverable performance, and a version using only ten percent of target learners recovered most of it. The practical recipe: deploying on a new course needs a cheap group-level recalibration, not a full refit.",
]));

children.push(...slide("Slide 7 — Difficulty geometry in a latent space", "1 minute, 15 seconds", [
  "The transfer result raises a deeper question: which courses are close to each other, and why? We fit a joint model over all three courses — global coefficients plus shrunken per-course deviation blocks, a partial-pooling structure across courses — and measured the similarity of the deviation vectors.",
  "Two of our courses form a mirror pair: English speakers learning Spanish, and Spanish speakers learning English — the same language contrast seen from opposite sides. Two courses share a learner population: the Spanish and French courses are both taken by English speakers.",
  "The geometry is unambiguous. The mirror pair is by far the closest pair in deviation space, with a cosine of minus 0.18 against a chance baseline of roughly minus 0.5, and with session and pacing effects strongly positively aligned. The shared-population pair sits at chance-level similarity, and the French course is the outlier with the largest deviation norm overall.",
  "In other words, difficulty structure follows the language contrast being learned, not the population doing the learning. A predictive check confirms these deviations carry real signal: applying the wrong course's deviation block is worse than applying none at all.",
  "This also closes the loop with our uncertainty story: epistemic uncertainty rises about 4.6 times under cross-course application, so the model's own uncertainty flags exactly the regime where its difficulty map needs re-estimation.",
]));

children.push(...slide("Slide 8 — The model ladder", "1 minute, 25 seconds", [
  "Finally, we asked where the remaining predictive signal lives. The hierarchical model has a natural Item Response Theory reading — learner ability against difficulty — and taking that reading seriously suggests one structural addition it lacked: difficulty parameters for individual words.",
  "That single addition is the largest improvement in the entire project: word-level item difficulties add about 0.04 AUROC on every course — four times any other extension — and, notably, land the extended model almost exactly at our official pipeline's level on all three courses.",
  "Three further additions each contribute smaller but reproducible gains: a leakage-safe dynamic-ability signal summarizing the learner's recent error history; memory features in the spirit of Duolingo's half-life regression — prior exposures to a word, the learner's error history on it, and the time since last exposure; and learner-by-format random slopes allowing individual differences in modality difficulty.",
  "Crucially, these passed the same forward-validation gate the HMM failed. Item difficulty improved AUROC by 0.036 and the full ladder by 0.059 relative to the current model, with better negative log likelihood and Brier score in every out-of-time fold. The dynamic-ability result resolves our HMM finding cleanly: learner state does carry signal, but it drifts continuously rather than jumping between discrete stages.",
  "One more discipline check: a gradient-boosted tree ensemble trained on the identical information performed no better — worse on two of three courses — so the interpretable additive structure costs nothing predictively. Every extension is a MAP estimate of a hierarchical prior and ports directly to the fully Bayesian pipeline.",
]));

children.push(...slide("Slide 9 — Routing with a guarantee", "1 minute, 10 seconds", [
  "This brings us to the deployment story: what should a tutor actually do with these predictions?",
  "The risk–coverage curve answers that. Sort tokens by predicted error probability, auto-handle the confident ones, and route the rest to review. Two layers make this trustworthy.",
  "The Bayesian layer chooses the routing threshold from posterior uncertainty, using the aleatoric–epistemic decomposition: inherent response variability versus insufficient evidence. As we saw, epistemic uncertainty is exactly what spikes under distribution shift, so the model defers where it should.",
  "The conformal layer adds something the posterior alone cannot: a distribution-free, finite-sample guarantee. Choosing the threshold on a learner-level calibration split guarantees the error rate among auto-handled tokens stays under a target budget, regardless of whether the model is well specified. The guarantee held almost exactly on held-out learners: at a five percent error budget the system safely auto-handles about half of all tokens, and at ten percent about 83 percent.",
  "The practical statement: half the workload can be automated with a provable one-in-twenty error ceiling, and everything the model is unsure about goes to a human — which is precisely the behavior we set out to build.",
]));

children.push(...slide("Slide 10 — What we learned", "1 minute", [
  "Four takeaways.",
  "First, hierarchy plus item difficulty is the model. Partial pooling wins decisively, and the Item Response Theory view — ability against difficulty — captures the dominant remaining signal on every language course.",
  "Second, negative results are results. Discrete acquisition stages are interpretable but add no forward-validated prediction, while continuous ability drift does — and our predefined validation gate is what told the difference.",
  "Third, transfer gaps are concept shift. The difficulty map changes across courses, follows the language contrast rather than the learner population, and is mostly erasable with a cheap group-level recalibration.",
  "Fourth, uncertainty is the product. Calibrated posteriors plus conformal guarantees turn a classifier into a tutor that provably knows when to hand off to a human.",
  "Next steps beyond the course: dynamic Item Response Theory with a state-space ability model, a full PyMC port of the item and slope effects, and explicit forgetting models. Thank you — we're happy to take questions.",
]));

const doc = new Document({
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 } } }, children }],
});
Packer.toBuffer(doc).then(b => { require("fs").writeFileSync("script_slides_6-10.docx", b); console.log("written"); });
