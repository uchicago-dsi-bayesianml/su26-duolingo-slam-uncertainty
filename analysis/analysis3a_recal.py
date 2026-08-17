"""Analysis 3a: recalibration-only transfer.

Freeze the fr_en model's shared coefficients; carry its linear predictor z_fr
into es_en as a single feature, then refit progressively richer target-side
adjustments on es_en TRAIN, evaluating on es_en dev:

  T0: raw transfer (z_fr as-is)                       [no target data]
  T1: group-intercept recal: z_fr + format/pos/dep/morph/country intercepts
  T2: T1 + learner intercepts                          [full target info]
  T1s/T2s: same, fit on a 10% learner slice of es_en train (low-resource)

Reference points: native es_en model (0.7145), native global-only (0.6699),
raw fr transfer (0.6220). If T1 recovers most of the native-global-only level,
the mismatch lives in group-level effect sizes, not model structure.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fit_models import build_vocab, fit_track, shared_matrix, learner_matrix

rng = np.random.default_rng(42)
tracks = {t: pd.read_parquet(f"slam/{t}_train.parquet") for t in ["fr_en", "es_en", "en_es"]}
dev = pd.read_parquet("slam/es_en_dev.parquet")
vocab = build_vocab(list(tracks.values()) + [dev])

# 1. fit fr_en, get frozen z on es_en train + dev
clf_fr, users_fr, ns = fit_track(tracks["fr_en"], vocab)
w = clf_fr.coef_.ravel()[:ns]
b0 = clf_fr.intercept_[0]
es_tr = tracks["es_en"]
z_tr = shared_matrix(es_tr, vocab) @ w + b0
z_dev = shared_matrix(dev, vocab) @ w + b0
y_tr, y_dev = es_tr["label"].values, dev["label"].values
print("T0 raw transfer AUROC:", round(roc_auc_score(y_dev, z_dev), 4), flush=True)

def cat_blocks(df, cols):
    blocks = []
    for c in cols:
        m = vocab[c]
        col = df[c].map(m)
        ok = col.notna().values
        rows = np.where(ok)[0]
        blocks.append(sp.csr_matrix((np.ones(ok.sum()), (rows, col[ok].astype(int).values)),
                                    shape=(len(df), len(m))))
    # morph
    mv = vocab["_morph"]
    r, cc = [], []
    for i, s in enumerate(df["morph"].values):
        if s:
            for t in s.split("|"):
                j = mv.get(t)
                if j is not None:
                    r.append(i); cc.append(j)
    blocks.append(sp.csr_matrix((np.ones(len(r)), (r, cc)), shape=(len(df), len(mv))))
    return blocks

GCOLS = ["client", "session", "format", "pos", "dep", "country"]

def tier(train_df, z_train, with_learner, label):
    zb = sp.csr_matrix(z_train.reshape(-1, 1))
    blocks = [zb] + cat_blocks(train_df, GCOLS)
    users = sorted(train_df["user"].unique())
    if with_learner:
        blocks.append(learner_matrix(train_df, users))
    X = sp.hstack(blocks).tocsr()
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(X, train_df["label"].values)
    dblocks = [sp.csr_matrix(z_dev.reshape(-1, 1))] + cat_blocks(dev, GCOLS)
    if with_learner:
        dblocks.append(learner_matrix(dev, users))
    Xd = sp.hstack(dblocks).tocsr()
    p = clf.predict_proba(Xd)[:, 1]
    auc = roc_auc_score(y_dev, p)
    print(f"{label}: AUROC {auc:.4f} (z-coef {clf.coef_.ravel()[0]:.3f})", flush=True)
    return auc

tier(es_tr, z_tr, False, "T1 group-recal (full es_en train)")
tier(es_tr, z_tr, True, "T2 group+learner recal (full)")

# low-resource: 10% of learners
users = es_tr["user"].unique()
sub_users = set(rng.choice(users, size=len(users) // 10, replace=False))
mask = es_tr["user"].isin(sub_users).values
tier(es_tr[mask].reset_index(drop=True), z_tr[mask], False, "T1s group-recal (10% learners)")
