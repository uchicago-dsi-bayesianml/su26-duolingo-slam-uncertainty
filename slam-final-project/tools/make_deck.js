const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

// palette
const NAVY = "0F2A4A";      // dark bg
const BLUE = "2A78D6";      // primary (matches figures)
const DEEPBLUE = "1C5CAB";
const ORANGE = "EB6834";    // accent (matches figures)
const AQUA = "1BAF7A";
const INK = "1A1A1A";
const INK2 = "52514E";
const MUT = "898781";
const TINT = "EEF4FC";      // light blue card tint
const TINT2 = "FDF1EC";     // light orange tint
const WHITE = "FFFFFF";

const W = 13.33, H = 7.5;

function title(slide, txt, sub) {
  slide.addText(txt, { x: 0.6, y: 0.35, w: W - 1.2, h: 0.75, fontSize: 30, bold: true,
    color: INK, fontFace: "Calibri", margin: 0 });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.02, w: W - 1.2, h: 0.4, fontSize: 14,
    color: INK2, fontFace: "Calibri", margin: 0 });
}
function pageno(slide, n) {
  slide.addText(String(n), { x: W - 0.75, y: H - 0.5, w: 0.4, h: 0.3, fontSize: 10,
    color: MUT, align: "right", fontFace: "Calibri", margin: 0 });
}

// ---------------- Slide 1: Title (dark) ----------------
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("Uncertainty-Aware Modeling of\nSecond-Language Acquisition", {
  x: 0.9, y: 1.5, w: 11.5, h: 1.9, fontSize: 40, bold: true, color: WHITE,
  fontFace: "Calibri", margin: 0, lineSpacingMultiple: 1.05 });
s.addText("A tutor that knows when it doesn't know.", {
  x: 0.9, y: 3.5, w: 11.5, h: 0.5, fontSize: 20, italic: true, color: "9EC5F4",
  fontFace: "Calibri", margin: 0 });
s.addText([
  { text: "Shane Dunkle  ·  Richard Pollitt", options: { fontSize: 16, color: WHITE, bold: true, breakLine: true } },
  { text: "ADSP 32014 — Bayesian Machine Learning with Generative AI Applications  ·  Summer 2026", options: { fontSize: 12.5, color: "C9D8EC" } },
], { x: 0.9, y: 5.7, w: 11.5, h: 0.9, fontFace: "Calibri", margin: 0, paraSpaceAfter: 6 });
s.addText("Duolingo SLAM 2018  ·  Bayes nets · hierarchical models · HMMs · MCMC · uncertainty & decision theory", {
  x: 0.9, y: 6.75, w: 11.5, h: 0.35, fontSize: 11, color: "7F97B5", fontFace: "Calibri", margin: 0 });
s.addNotes("Two-person project. One sentence framing: adaptive tutors must decide not just whether a learner will err, but when to trust that prediction. We built a Bayesian pipeline that outputs calibrated probabilities and an explicit route-to-human option.");

// ---------------- Slide 2: Problem & data ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "The problem: error prediction that admits what it doesn't know",
  "Duolingo SLAM 2018 — public corpus of learner exercise responses over each learner's first 30 days");
const stats = [
  ["5.5M+", "exercise tokens across three courses", BLUE],
  ["6,400+", "learners (es_en, fr_en, en_es tracks)", BLUE],
  ["~15%", "of tokens are errors — the minority class we predict", ORANGE],
  ["30 days", "per learner; dev/test are later in time than train", BLUE],
];
stats.forEach((st, i) => {
  const x = 7.15, y = 1.7 + i * 1.32;
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 5.5, h: 1.12, fill: { color: i === 2 ? TINT2 : TINT }, rectRadius: 0.08, line: { type: "none" } });
  s.addText(st[0], { x: x + 0.25, y: y + 0.08, w: 1.75, h: 0.95, fontSize: 30, bold: true, color: st[2], fontFace: "Calibri", margin: 0, valign: "middle" });
  s.addText(st[1], { x: x + 2.05, y: y + 0.08, w: 3.3, h: 0.95, fontSize: 12.5, color: INK2, fontFace: "Calibri", margin: 0, valign: "middle" });
});
s.addText([
  { text: "Point predictions aren't enough.", options: { bold: true, fontSize: 16, color: INK, breakLine: true } },
  { text: "A standard classifier outputs one probability and hides its own uncertainty. An adaptive tutor acting on a wrong, confident prediction wastes learner time — or worse, misdiagnoses them.", options: { fontSize: 14, color: INK2, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Our reframing: Bayesian inference under uncertainty.", options: { bold: true, fontSize: 16, color: INK, breakLine: true } },
  { text: "Predict per-token error probability with calibrated uncertainty, decompose why the model is unsure (inherent difficulty vs. not enough data), and let a decision layer abstain and route to a human when confidence is low.", options: { fontSize: 14, color: INK2, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Rich features per token: learner, L1/country, days-in-course, exercise format, session type, response time, POS, morphology, dependency structure.", options: { fontSize: 12, italic: true, color: MUT } },
], { x: 0.6, y: 1.75, w: 6.15, h: 5.1, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4, valign: "top" });
pageno(s, 2);
s.addNotes("Emphasize the dataset is public (Harvard Dataverse) and the shared-task benchmark exists (AUROC ~0.86 for top 2018 systems with heavy engineering). Errors are the minority class, so we report AUROC and calibration, never accuracy.");

// ---------------- Slide 3: Approach ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "Three coupled Bayesian components", "Each maps directly onto the course syllabus");
const cards = [
  ["A — Hierarchical Bayesian error model", BLUE,
   "Multilevel logistic regression with partial pooling: learners nested within L1 groups and skill/format types. Sparse learners borrow strength; group effects expose L1 transfer and difficulty.", "Bayes nets · hierarchical models · MCMC (NUTS) · ADVI"],
  ["B — Latent acquisition stages (HMM)", AQUA,
   "Per-learner Hidden Markov Model over the 30-day timeline: latent states as acquisition stages (unlearned → learning → acquired), tested for predictive value under forward validation.", "Markov models · HMMs"],
  ["C — Uncertainty & routing", ORANGE,
   "Aleatoric vs. epistemic decomposition from hierarchical posterior draws; a Bayesian utility layer abstains and routes to a human when uncertainty is high. Evaluated with risk–coverage curves.", "Decision theory · Bayesian deep learning ideas"],
];
cards.forEach((c, i) => {
  const x = 0.6 + i * 4.15, y = 1.8;
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 3.9, h: 4.5, fill: { color: "F7F9FC" }, rectRadius: 0.1, line: { color: "E3E8F0", width: 1 } });
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fill: { color: c[1] }, line: { type: "none" } });
  s.addText(["A", "B", "C"][i], { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, align: "center", valign: "middle", fontSize: 20, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
  s.addText(c[0].slice(4), { x: x + 0.3, y: y + 1.05, w: 3.3, h: 0.8, fontSize: 16.5, bold: true, color: INK, fontFace: "Calibri", margin: 0 });
  s.addText(c[2], { x: x + 0.3, y: y + 1.9, w: 3.3, h: 1.9, fontSize: 12.5, color: INK2, fontFace: "Calibri", margin: 0 });
  s.addText(c[3], { x: x + 0.3, y: y + 3.9, w: 3.3, h: 0.5, fontSize: 10.5, italic: true, color: MUT, fontFace: "Calibri", margin: 0 });
});
s.addText("Final predictive model: the hierarchical Bayesian logistic regression (A). B became a descriptive model; C became the deployment story.",
  { x: 0.6, y: 6.55, w: 12.1, h: 0.5, fontSize: 13, italic: true, color: INK2, fontFace: "Calibri", margin: 0 });
pageno(s, 3);
s.addNotes("Division of labor: Richard — data engineering, scale, baselines, validation protocol. Shane — Bayesian modeling and SLA interpretation. Tools: PyMC/NumPyro, ArviZ, hmmlearn, scikit-learn.");

// ---------------- Slide 4: Core results ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "The hierarchical model earns its structure", "Native-fit performance on the official development sets");
const res = [
  ["0.755", "AUROC — es_en (English speakers learning Spanish)"],
  ["0.777", "AUROC — fr_en (learning French)"],
  ["0.777", "AUROC — en_es (learning English)"],
];
res.forEach((r, i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.8, w: 3.9, h: 1.7, fill: { color: TINT }, rectRadius: 0.1, line: { type: "none" } });
  s.addText(r[0], { x: x + 0.3, y: 1.95, w: 3.3, h: 0.85, fontSize: 40, bold: true, color: DEEPBLUE, fontFace: "Calibri", margin: 0 });
  s.addText(r[1], { x: x + 0.3, y: 2.8, w: 3.3, h: 0.6, fontSize: 11.5, color: INK2, fontFace: "Calibri", margin: 0 });
});
s.addText([
  { text: "Partial pooling wins decisively over the pooled baseline.", options: { bold: true, fontSize: 16, color: INK, breakLine: true } },
  { text: "ELPD improvement +2,158 ± 77 (PSIS-LOO) — about 28 standard errors. Calibration is strong: ECE 0.014 vs 0.020 pooled, with better Brier and NLL across the board.", options: { fontSize: 14, color: INK2, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Interpretable by construction.", options: { bold: true, fontSize: 16, color: INK, breakLine: true } },
  { text: "Group-level posteriors expose which exercise formats are hard, which skills trip up which learner populations, and how much individual learners deviate — the SLA story falls straight out of the model's parameters.", options: { fontSize: 14, color: INK2 } },
], { x: 0.6, y: 3.85, w: 7.6, h: 3.1, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4 });
s.addShape(pres.ShapeType.roundRect, { x: 8.5, y: 3.85, w: 4.2, h: 2.9, fill: { color: "F7F9FC" }, rectRadius: 0.1, line: { color: "E3E8F0", width: 1 } });
s.addText([
  { text: "Reference points", options: { bold: true, fontSize: 13.5, color: INK, breakLine: true } },
  { text: "Tuned non-Bayesian logistic baseline: AUROC 0.814 (heavier feature set)", options: { fontSize: 11.5, color: INK2, bullet: true, breakLine: true } },
  { text: "2018 shared-task top systems: ≈ 0.86 with extensive engineering", options: { fontSize: 11.5, color: INK2, bullet: true, breakLine: true } },
  { text: "Our goal: calibrated uncertainty + interpretability at competitive discrimination", options: { fontSize: 11.5, color: INK2, bullet: true } },
], { x: 8.75, y: 4.05, w: 3.7, h: 2.5, fontFace: "Calibri", margin: 0, paraSpaceAfter: 6 });
pageno(s, 4);
s.addNotes("ELPD numbers from the LOO comparison (run_summary.json). If asked about the gap to 0.86: those systems used massive feature engineering and ensembling without uncertainty; our extended ladder (slide 8) closes much of the gap while staying interpretable.");

// ---------------- Slide 5: HMM negative result ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "An honest negative result: acquisition stages don't add prediction",
  "Three-state HMM per learner — interpretable, but gated out by forward validation");
s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.85, w: 6.0, h: 4.4, fill: { color: "F2FAF6" }, rectRadius: 0.1, line: { color: "D5EDE2", width: 1 } });
s.addText([
  { text: "What the HMM found ✓", options: { bold: true, fontSize: 17, color: "0E7A55", breakLine: true } },
  { text: "Interpretable unlearned → learning → acquired states over each learner's 30-day timeline", options: { fontSize: 13, color: INK2, bullet: true, breakLine: true } },
  { text: "Stage transitions track known difficulty and pacing patterns", options: { fontSize: 13, color: INK2, bullet: true, breakLine: true } },
  { text: "Kept in the report as a descriptive latent-learning model", options: { fontSize: 13, color: INK2, bullet: true } },
], { x: 0.9, y: 2.1, w: 5.4, h: 3.9, fontFace: "Calibri", margin: 0, paraSpaceAfter: 8, valign: "top" });
s.addShape(pres.ShapeType.roundRect, { x: 6.85, y: 1.85, w: 6.0, h: 4.4, fill: { color: TINT2 }, rectRadius: 0.1, line: { color: "F2DCCF", width: 1 } });
s.addText([
  { text: "What it didn't ✗", options: { bold: true, fontSize: 17, color: "B44A1F", breakLine: true } },
  { text: "Ungated hybrid significantly worsened NLL (+0.00055, 95% CI [+0.00018, +0.00092]) and Brier — in all four forward folds", options: { fontSize: 13, color: INK2, bullet: true, breakLine: true } },
  { text: "Entropy-gated hybrid statistically indistinguishable from the logistic baseline", options: { fontSize: 13, color: INK2, bullet: true, breakLine: true } },
  { text: "The small dev-set 'gain' was selection optimism — exactly what the gate exists to catch", options: { fontSize: 13, color: INK2, bullet: true } },
], { x: 7.15, y: 2.1, w: 5.4, h: 3.9, fontFace: "Calibri", margin: 0, paraSpaceAfter: 8, valign: "top" });
s.addText("Takeaway: learner state matters, but it drifts continuously rather than jumping between discrete stages — not every interpretable Bayesian structure needs to improve prediction.",
  { x: 0.6, y: 6.5, w: 12.1, h: 0.6, fontSize: 14, italic: true, bold: true, color: INK, fontFace: "Calibri", margin: 0 });
pageno(s, 5);
s.addNotes("This is a strength, not a weakness: we predefined a confirmation gate and it correctly failed. Foreshadow slide 8: a continuous dynamic-ability signal DOES pass the same gate — the dynamics are real, the discreteness was wrong.");

// ---------------- Slide 6: Transfer ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "Cross-course transfer: the gap is real — and diagnosable");
s.addImage({ path: "fig1_transfer_matrix.png", x: 0.6, y: 1.75, w: 6.1, h: 4.7 });
s.addText([
  { text: "The penalty is reproducible.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "Learner-clustered bootstrap on the model's own posterior draws: transfer AUROC 0.732, 95% CI [0.727, 0.737] — native 0.755 sits far outside. Refit variance is 5× smaller than the gap.", options: { fontSize: 13, color: INK2, breakLine: true } },
  { text: "The ranking is not.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "Which source transfers worst (fr_en vs en_es) lies within sampling noise — we report the penalty, not a ranking.", options: { fontSize: 13, color: INK2, breakLine: true } },
  { text: "It's concept shift, not covariate shift.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "Importance-weighting the source data toward the target feature distribution makes transfer worse. Refitting only group-level intercepts on ~10% of target learners recovers most native performance: the difficulty map shifts, the model structure transfers.", options: { fontSize: 13, color: INK2 } },
], { x: 7.05, y: 1.85, w: 5.7, h: 5.3, fontFace: "Calibri", margin: 0, paraSpaceAfter: 7, valign: "top" });
pageno(s, 6);
s.addNotes("Practical recipe: deploying on a new course needs only a cheap group-intercept recalibration on a small slice of target data, not a full refit. fr_en and es_en share the same L1-English population (only 2 shared users) yet still show the gap — it's about the course, not the people.");

// ---------------- Slide 7: Latent space ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "Difficulty geometry follows the language contrast, not the learner",
  "Joint model over all three tracks: global effects + shrunk per-track deviations (partial pooling across courses)");
// map visual: es_en & en_es close, fr_en far
const cy = 4.1;
s.addShape(pres.ShapeType.line, { x: 4.05, y: cy - 0.65, w: 1.5, h: 0.0, line: { color: "B9C6D8", width: 1.5, dashType: "dash" } });
s.addShape(pres.ShapeType.line, { x: 2.0, y: cy - 0.4, w: 1.15, h: 1.45, line: { color: "E3C4B4", width: 1.5, dashType: "dash" } });
s.addShape(pres.ShapeType.line, { x: 3.4, y: cy + 1.15, w: 1.7, h: -1.5, line: { color: "E3C4B4", width: 1.5, dashType: "dash" } });
s.addShape(pres.ShapeType.ellipse, { x: 3.35, y: cy - 1.15, w: 1.1, h: 1.1, fill: { color: BLUE }, line: { type: "none" } });
s.addText("es_en", { x: 3.35, y: cy - 1.15, w: 1.1, h: 1.1, align: "center", valign: "middle", fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
s.addShape(pres.ShapeType.ellipse, { x: 5.1, y: cy - 1.05, w: 1.1, h: 1.1, fill: { color: AQUA }, line: { type: "none" } });
s.addText("en_es", { x: 5.1, y: cy - 1.05, w: 1.1, h: 1.1, align: "center", valign: "middle", fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
s.addShape(pres.ShapeType.ellipse, { x: 1.15, y: cy + 0.75, w: 1.1, h: 1.1, fill: { color: ORANGE }, line: { type: "none" } });
s.addText("fr_en", { x: 1.15, y: cy + 0.75, w: 1.1, h: 1.1, align: "center", valign: "middle", fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
s.addText("mirror pair: cosine −0.18\n(chance ≈ −0.5)", { x: 3.7, y: cy - 2.0, w: 2.8, h: 0.7, fontSize: 10.5, color: INK2, fontFace: "Calibri", margin: 0, align: "center" });
s.addText("fr_en: the outlier —\nlargest deviation norm", { x: 0.7, y: cy + 2.0, w: 2.6, h: 0.6, fontSize: 10.5, color: INK2, fontFace: "Calibri", margin: 0 });
s.addText("Deviation-space similarity (schematic)", { x: 0.7, y: 6.6, w: 5.5, h: 0.35, fontSize: 10, italic: true, color: MUT, fontFace: "Calibri", margin: 0 });
s.addText([
  { text: "The mirror pair clusters.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "es_en and en_es — the same Spanish↔English contrast seen from opposite banks — are by far the closest tracks in deviation space (cosine −0.18 vs a −0.5 chance baseline), with session and pacing effects strongly positively aligned (+0.83, +0.53).", options: { fontSize: 13, color: INK2, breakLine: true } },
  { text: "Shared learners don't cluster.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "fr_en and es_en share the same L1-English learner population, yet sit at chance-level similarity (−0.51). What transfers is the structure of the language contrast, not the population.", options: { fontSize: 13, color: INK2, breakLine: true } },
  { text: "Wrong-track adaptation is worse than none.", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "Applying another course's deviation block underperforms the global model — track deviations carry real, course-specific signal.", options: { fontSize: 13, color: INK2 } },
], { x: 7.0, y: 1.9, w: 5.75, h: 5.1, fontFace: "Calibri", margin: 0, paraSpaceAfter: 7 });
pageno(s, 7);
s.addNotes("This is the quantitative version of Prof. Gundogdu's 'languages as points in a latent space' remark from our topic-selection email — a publishable-flavored SLA finding. Cosine baseline is ~-0.5 because three deviation blocks approximately sum to zero under shrinkage.");

// ---------------- Slide 8: Model ladder ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "Where the remaining signal lives: an item-response ladder");
s.addImage({ path: "fig3_model_ladder.png", x: 0.55, y: 1.75, w: 7.0, h: 4.5 });
s.addShape(pres.ShapeType.roundRect, { x: 7.85, y: 1.8, w: 4.9, h: 1.95, fill: { color: TINT }, rectRadius: 0.1, line: { type: "none" } });
s.addText([
  { text: "+0.04 AUROC", options: { bold: true, fontSize: 26, color: DEEPBLUE, breakLine: true } },
  { text: "from word-level item difficulty alone (Bayesian IRT view: ability − difficulty). The dominant gain on every track — 4× any other addition.", options: { fontSize: 12, color: INK2 } },
], { x: 8.1, y: 1.95, w: 4.4, h: 1.7, fontFace: "Calibri", margin: 0, paraSpaceAfter: 3 });
s.addText([
  { text: "Passed the same forward-validation gate the HMM failed:", options: { bold: true, fontSize: 13.5, color: INK, breakLine: true } },
  { text: "IRT: +0.036 AUROC, −0.017 NLL — better in every out-of-time fold", options: { fontSize: 12.5, color: INK2, bullet: true, breakLine: true } },
  { text: "Full ladder (+ dynamic ability, memory/HLR, learner×format slopes): +0.059 AUROC, −0.030 NLL", options: { fontSize: 12.5, color: INK2, bullet: true, breakLine: true } },
  { text: "Continuous ability drift passes where discrete stages failed", options: { fontSize: 12.5, color: INK2, bullet: true, breakLine: true } },
  { text: "The interpretable GLM matches or beats gradient-boosted trees on the same features — nonlinearity buys no consistent gain", options: { fontSize: 12.5, color: INK2, bullet: true } },
], { x: 7.85, y: 4.0, w: 4.9, h: 3.0, fontFace: "Calibri", margin: 0, paraSpaceAfter: 6 });
pageno(s, 8);
s.addNotes("All extensions are MAP estimates of hierarchical priors and port directly to PyMC (item difficulties = item-level random effects; slopes = LKJ-correlated random effects). Every rung is significant under a learner-clustered bootstrap; the ladder replicates on all three language tracks.");

// ---------------- Slide 9: Routing ----------------
s = pres.addSlide();
s.background = { color: WHITE };
title(s, "Knowing its limits — with a guarantee");
s.addImage({ path: "fig2_risk_coverage.png", x: 0.55, y: 1.8, w: 6.9, h: 4.75 });
const conf = [
  ["50%", "of tokens auto-handled at a 5% error budget", BLUE],
  ["83%", "auto-handled at a 10% error budget", BLUE],
];
conf.forEach((c, i) => {
  const y = 1.9 + i * 1.5;
  s.addShape(pres.ShapeType.roundRect, { x: 7.8, y, w: 4.95, h: 1.3, fill: { color: TINT }, rectRadius: 0.09, line: { type: "none" } });
  s.addText(c[0], { x: 8.05, y: y + 0.1, w: 1.5, h: 1.1, fontSize: 34, bold: true, color: DEEPBLUE, fontFace: "Calibri", margin: 0, valign: "middle" });
  s.addText(c[1], { x: 9.6, y: y + 0.1, w: 3.0, h: 1.1, fontSize: 12.5, color: INK2, fontFace: "Calibri", margin: 0, valign: "middle" });
});
s.addText([
  { text: "Two layers of trust:", options: { bold: true, fontSize: 14, color: INK, breakLine: true } },
  { text: "Bayesian utility layer picks the routing threshold from posterior uncertainty (aleatoric vs. epistemic — epistemic rises 4.6× under cross-course shift, flagging exactly where to defer)", options: { fontSize: 12, color: INK2, bullet: true, breakLine: true } },
  { text: "Split-conformal layer adds a distribution-free, finite-sample guarantee on the error rate among auto-handled tokens — it held within 0.3pp on held-out learners", options: { fontSize: 12, color: INK2, bullet: true } },
], { x: 7.8, y: 5.05, w: 4.95, h: 2.1, fontFace: "Calibri", margin: 0, paraSpaceAfter: 6 });
pageno(s, 9);
s.addNotes("This is the deployment story: the tutor handles the easy half provably safely and routes the rest for review. Conformal is assumption-free, so it protects against miscalibration of the posterior itself.");

// ---------------- Slide 10: Takeaways (dark) ----------------
s = pres.addSlide();
s.background = { color: NAVY };
s.addText("What we learned", { x: 0.9, y: 0.55, w: 11.5, h: 0.8, fontSize: 32, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
const tk = [
  ["1", "Hierarchy + item difficulty is the model.", "Partial pooling wins decisively, and an IRT reading (ability − difficulty) captures the dominant remaining signal on every language track."],
  ["2", "Negative results are results.", "Discrete acquisition stages are interpretable but add no forward-validated prediction; continuous ability drift does. The predefined gate caught the difference."],
  ["3", "Transfer gaps are concept shift.", "The difficulty map changes across courses — and follows the language contrast, not the learner population. A cheap group-level recalibration mostly erases it."],
  ["4", "Uncertainty is the product.", "Calibrated posteriors + conformal guarantees turn a classifier into a tutor that provably knows when to hand off to a human."],
];
tk.forEach((t, i) => {
  const x = 0.9 + (i % 2) * 6.05, y = 1.65 + Math.floor(i / 2) * 2.15;
  s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: ORANGE }, line: { type: "none" } });
  s.addText(t[0], { x, y: y + 0.05, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
  s.addText([
    { text: t[1], options: { bold: true, fontSize: 15.5, color: WHITE, breakLine: true } },
    { text: t[2], options: { fontSize: 12, color: "C9D8EC" } },
  ], { x: x + 0.7, y, w: 5.1, h: 2.0, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4 });
});
s.addText("Next: dynamic IRT (state-space ability), full PyMC port of item effects and correlated slopes, half-life forgetting models — and a write-up beyond the course.",
  { x: 0.9, y: 6.35, w: 11.5, h: 0.5, fontSize: 13, italic: true, color: "9EC5F4", fontFace: "Calibri", margin: 0 });
s.addText("Shane Dunkle · Richard Pollitt — thank you", { x: 0.9, y: 6.95, w: 11.5, h: 0.4, fontSize: 12, color: "7F97B5", fontFace: "Calibri", margin: 0 });
s.addNotes("Close on the publication angle: the instructor flagged this topic as publishable; the latent-space finding plus the IRT ladder is the core of that paper.");

pres.writeFile({ fileName: "SLAM_Final_Presentation.pptx" }).then(() => console.log("written"));
