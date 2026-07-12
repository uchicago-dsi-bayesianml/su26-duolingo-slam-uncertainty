# SLAM — Uncertainty-Aware Modeling of Second-Language Acquisition

**Bayesian ML final project · ADSP 32014/ON01 · Summer 2026 · Shane Dunkle & Richard Pollitt**

Predict **when a language learner will make an error** on Duolingo exercises — *with calibrated uncertainty*, so the model can say **"I don't know, route to a human / adaptive review"** instead of guessing.

## 📊 Live tracker
An interactive task tracker ships as [`index.html`](index.html). After enabling **GitHub Pages** (Settings → Pages → deploy from `main`, root), it will be live at:
`https://<your-username>.github.io/<repo-name>/`

## 🧪 Approach
- **Part A — Hierarchical Bayesian error model.** Learners nested within native-language (L1) groups and skill/format types; partial pooling; fit by MCMC (NUTS) / ADVI.
- **Part B — HMM acquisition stages.** Latent learning stages over each learner's first 30 days.
- **Part C — Bayesian neural network.** Aleatoric vs. epistemic uncertainty → decision-theoretic abstention / route-to-human.

Covers Bayes nets, hierarchical models, HMMs, MCMC, decision theory, and Bayesian neural networks.

## 📦 Data
Duolingo **SLAM 2018** — Harvard Dataverse (`doi:10.7910/DVN/8SWHNO`) · shared task: https://sharedtask.duolingo.com/2018.html
~2M+ tokens, 6,000+ learners, first 30 days, English/Spanish/French. Top shared-task AUROC ≈ 0.86.
**Raw data is not committed** — see [`data/README.md`](data/README.md).

## 🗂️ Structure
```
index.html        Interactive project tracker (GitHub Pages)
data/             How to obtain SLAM data (not committed)
notebooks/        Analysis & modeling notebooks
src/              Data pipeline, models, evaluation
```

## 👥 Team
- **Shane Dunkle** — Bayesian modeling (hierarchical / MCMC / HMM / BNN) + SLA interpretation & write-up
- **Richard Pollitt** — data engineering, databases/SQL, scale & infrastructure, project management

## 🔗 Links
- Notion project plan: https://app.notion.com/p/39bf5c743c008175a58aec6cd755c283
- Google Drive (Final Project): https://drive.google.com/drive/folders/1jc7b-Q46RQEnVS2cU02Zlv5DR752zzpj

## 🗓️ Milestones
Subject **Jul 6** · Abstract **Jul 20** · Project **Aug 10** · Presentation **Aug 17**

---
*Coursework project for ADSP 32014, MS in Applied Data Science, University of Chicago.*
