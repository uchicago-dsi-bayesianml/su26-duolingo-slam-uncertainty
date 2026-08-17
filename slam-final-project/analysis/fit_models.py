"""Fit MAP hierarchical-logistic analogues per track; save es_en dev predictions.

Model: P(error) = sigmoid(B.x_shared + a_learner), L2-penalized (= MAP of
random-intercept logistic). Shared block uses a vocabulary common to all
tracks so coefficients transfer; learner block is track-specific and is
zeroed for cross-track transfer (matching the 'unknown learner' situation).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

CATS = ["client", "session", "format", "pos", "dep", "country"]

def build_vocab(dfs):
    vocab = {}
    for c in CATS:
        vals = sorted(set().union(*[set(d[c].unique()) for d in dfs]))
        vocab[c] = {v: i for i, v in enumerate(vals)}
    tags = set()
    for d in dfs:
        for s in d["morph"].unique():
            if s:
                tags.update(s.split("|"))
    vocab["_morph"] = {t: i for i, t in enumerate(sorted(tags))}
    return vocab

def shared_matrix(df, vocab):
    blocks = []
    n = len(df)
    for c in CATS:
        m = vocab[c]
        col = df[c].map(m)
        ok = col.notna().values
        rows = np.where(ok)[0]
        data = np.ones(ok.sum())
        blocks.append(sp.csr_matrix((data, (rows, col[ok].astype(int).values)),
                                    shape=(n, len(m))))
    # morph multi-hot
    mv = vocab["_morph"]
    rows_l, cols_l = [], []
    for i, s in enumerate(df["morph"].values):
        if s:
            for t in s.split("|"):
                j = mv.get(t)
                if j is not None:
                    rows_l.append(i); cols_l.append(j)
    blocks.append(sp.csr_matrix((np.ones(len(rows_l)), (rows_l, cols_l)),
                                shape=(n, len(mv))))
    days = np.log1p(df["days"].clip(lower=0)).values.reshape(-1, 1)
    t = df["time"].values.copy()
    tmiss = (t < 0).astype(float).reshape(-1, 1)
    t = np.where(t < 0, 0, t)
    logt = np.log1p(t).reshape(-1, 1)
    blocks.append(sp.csr_matrix(np.hstack([days, logt, tmiss])))
    return sp.hstack(blocks).tocsr()

def learner_matrix(df, users):
    m = {u: i for i, u in enumerate(users)}
    col = df["user"].map(m)
    ok = col.notna().values
    rows = np.where(ok)[0]
    return sp.csr_matrix((np.ones(ok.sum()), (rows, col[ok].astype(int).values)),
                         shape=(len(df), len(users)))

def fit_track(train, vocab, C=1.0, seed=0):
    users = sorted(train["user"].unique())
    X = sp.hstack([shared_matrix(train, vocab), learner_matrix(train, users)]).tocsr()
    y = train["label"].values
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(X, y)
    n_shared = X.shape[1] - len(users)
    return clf, users, n_shared

def predict(clf, users, n_shared, df, vocab, use_learner):
    Xs = shared_matrix(df, vocab)
    w = clf.coef_.ravel()
    z = Xs @ w[:n_shared] + clf.intercept_[0]
    if use_learner:
        Xl = learner_matrix(df, users)
        z = z + Xl @ w[n_shared:]
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    tracks = {}
    for t in ["fr_en", "es_en", "en_es"]:
        tracks[t] = pd.read_parquet(f"slam/{t}_train.parquet")
    dev = pd.read_parquet("slam/es_en_dev.parquet")
    vocab = build_vocab(list(tracks.values()) + [dev])

    fr_users = set(tracks["fr_en"].user.unique())
    es_users = set(tracks["es_en"].user.unique())
    enes_users = set(tracks["en_es"].user.unique())
    print("user overlap fr_en∩es_en:", len(fr_users & es_users),
          "| en_es∩es_en:", len(enes_users & es_users))

    preds = {"label": dev["label"].values, "user": dev["user"].values}
    for t in ["fr_en", "es_en", "en_es"]:
        clf, users, ns = fit_track(tracks[t], vocab)
        preds[f"p_{t}"] = predict(clf, users, ns, dev, vocab, use_learner=(t == "es_en"))
        if t == "es_en":
            preds["p_es_en_global"] = predict(clf, users, ns, dev, vocab, use_learner=False)
        from sklearn.metrics import roc_auc_score
        print(t, "-> es_en dev AUROC:",
              round(roc_auc_score(preds["label"], preds[f"p_{t}"]), 4), flush=True)
    from sklearn.metrics import roc_auc_score
    print("es_en global-only AUROC:",
          round(roc_auc_score(preds["label"], preds["p_es_en_global"]), 4))
    np.savez("slam/es_dev_preds.npz", **preds)
    print("saved slam/es_dev_preds.npz")
