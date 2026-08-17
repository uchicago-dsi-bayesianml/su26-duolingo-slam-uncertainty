"""Analysis 3b: covariate shift between fr_en and es_en.

(1) Per-feature total-variation distance between fr_en train and es_en dev
    marginals (which inputs actually shift?).
(2) Domain classifier (fr_en train vs es_en train) on shared features ->
    importance weights for fr_en rows -> refit fr_en model with weights ->
    transfer AUROC on es_en dev. If reweighting recovers the gap, it's
    covariate shift; if not, it's concept shift P(error|x).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fit_models import build_vocab, fit_track, shared_matrix, predict, CATS

rng = np.random.default_rng(3)
tracks = {t: pd.read_parquet(f"slam/{t}_train.parquet") for t in ["fr_en", "es_en", "en_es"]}
dev = pd.read_parquet("slam/es_en_dev.parquet")
vocab = build_vocab(list(tracks.values()) + [dev])
fr, es = tracks["fr_en"], tracks["es_en"]

print("=== TV distance of marginals: fr_en train vs es_en dev ===")
for c in CATS + ["morph_head"]:
    if c == "morph_head":
        a = fr["morph"].str.split("|").str[0].fillna("")
        b = dev["morph"].str.split("|").str[0].fillna("")
    else:
        a, b = fr[c], dev[c]
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    cats = set(pa.index) | set(pb.index)
    tv = 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in cats)
    print(f"  {c:12s} TV = {tv:.3f}")
for c in ["days", "time"]:
    qa = np.percentile(fr[c], [25, 50, 75])
    qb = np.percentile(dev[c], [25, 50, 75])
    print(f"  {c:12s} quartiles fr={np.round(qa,1)} es_dev={np.round(qb,1)}")

print("\n=== Domain classifier + importance-weighted refit ===", flush=True)
# subsample for the domain classifier
n_sub = 400_000
fr_s = fr.sample(min(n_sub, len(fr)), random_state=1)
es_s = es.sample(min(n_sub, len(es)), random_state=1)
dom = pd.concat([fr_s, es_s], ignore_index=True)
ydom = np.r_[np.zeros(len(fr_s)), np.ones(len(es_s))]
Xdom = shared_matrix(dom, vocab)
dclf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
dclf.fit(Xdom, ydom)
print("domain classifier AUROC (separability of tracks):",
      round(roc_auc_score(ydom, dclf.predict_proba(Xdom)[:, 1]), 4), flush=True)

# weights for ALL fr_en rows: w = p(es|x)/p(fr|x), clipped
pfr = dclf.predict_proba(shared_matrix(fr, vocab))[:, 1]
wgt = np.clip(pfr / np.clip(1 - pfr, 1e-6, None), 0, 20)
wgt *= len(wgt) / wgt.sum()
print("weight stats: median", round(np.median(wgt), 3), "p95", round(np.percentile(wgt, 95), 2),
      "ESS frac", round((wgt.sum()**2 / (len(wgt) * (wgt**2).sum())), 3), flush=True)

# weighted refit of fr_en
from fit_models import learner_matrix
users = sorted(fr["user"].unique())
X = sp.hstack([shared_matrix(fr, vocab), learner_matrix(fr, users)]).tocsr()
clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
clf.fit(X, fr["label"].values, sample_weight=wgt)
ns = X.shape[1] - len(users)
p = predict(clf, users, ns, dev, vocab, use_learner=False)
y = dev["label"].values
print("importance-weighted fr_en -> es_en dev AUROC:", round(roc_auc_score(y, p), 4))
print("(unweighted was 0.6220; native global-only 0.6699)")
