"""Forward-validation gate (Richard's protocol, out-of-time folds) on es_en.

Within each learner's chronological train sequence, define 3 forward folds by
position quantile: [0.7,0.8), [0.8,0.9), [0.9,1.0). For each fold: fit on all
tokens BEFORE the fold window, evaluate on the fold window. Gate M3 (IRT) and
M6 (full ladder) against M2 (current model analogue): consistent-sign deltas
in AUROC/NLL/Brier across folds.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from fit_models import build_vocab, shared_matrix, learner_matrix

tr = pd.read_parquet("slam/es_en_train.parquet")
ftr = pd.read_parquet("slam/es_en_train_seqfeat.parquet")
dv = pd.read_parquet("slam/es_en_dev.parquet")
vocab = build_vocab([tr, dv])

# within-user position fraction (train is chronological per user)
pos = tr.groupby("user", sort=False).cumcount().values
cnt = tr.groupby("user", sort=False)["label"].transform("size").values
frac = pos / cnt

users = sorted(tr["user"].unique())
umap = {u: i for i, u in enumerate(users)}
wc = tr["token"].value_counts()
wmap = {w: i for i, w in enumerate(sorted(wc[wc >= 20].index))}

def word_block(df):
    col = df["token"].map(wmap)
    ok = col.notna().values
    rows = np.where(ok)[0]
    return sp.csr_matrix((np.ones(ok.sum()), (rows, col[ok].astype(int).values)),
                         shape=(len(df), len(wmap)))

def seq_block(f):
    g = np.log1p(np.clip(f["gap_days"].values, 0, None))
    never = (f["gap_days"].values < 0).astype(float)
    return sp.csr_matrix(np.column_stack([
        f["dyn_ab"].values, f["cum_err"].values, f["pos"].values,
        f["seen"].values, f["tok_err"].values, g, never, f["first"].values]))

def slope_block(df):
    fi = df["format"].map(vocab["format"]).values
    ui = df["user"].map(umap)
    ok = ui.notna().values
    rows = np.where(ok)[0]
    cols = (ui[ok].astype(int).values * 3 + fi[ok]).astype(int)
    return sp.csr_matrix((0.5 * np.ones(ok.sum()), (rows, cols)),
                         shape=(len(df), len(users) * 3))

def design(df, f, model):
    blocks = [shared_matrix(df, vocab), learner_matrix(df, users)]
    if model in ("M3", "M6"):
        blocks.append(word_block(df))
    if model == "M6":
        blocks.append(seq_block(f))
        blocks.append(slope_block(df))
    return sp.hstack(blocks).tocsr()

FOLDS = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
res = {m: [] for m in ["M2", "M3", "M6"]}
for lo, hi in FOLDS:
    fit_m = frac < lo
    ev_m = (frac >= lo) & (frac < hi)
    dtr, dft = tr[fit_m].reset_index(drop=True), ftr[fit_m].reset_index(drop=True)
    dev_, fev = tr[ev_m].reset_index(drop=True), ftr[ev_m].reset_index(drop=True)
    yf, ye = dtr["label"].values, dev_["label"].values
    for m in ["M2", "M3", "M6"]:
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=600, tol=1e-4)
        clf.fit(design(dtr, dft, m), yf)
        p = np.clip(clf.predict_proba(design(dev_, fev, m))[:, 1], 1e-6, 1 - 1e-6)
        r = (roc_auc_score(ye, p), log_loss(ye, p), brier_score_loss(ye, p))
        res[m].append(r)
        print(f"fold [{lo},{hi}) {m}: AUROC {r[0]:.4f} NLL {r[1]:.4f} Brier {r[2]:.4f}",
              flush=True)

print("\n=== GATE SUMMARY (deltas vs M2, per fold) ===")
for m in ["M3", "M6"]:
    d = np.array(res[m]) - np.array(res["M2"])
    for i, met in enumerate(["AUROC", "NLL", "Brier"]):
        vals = d[:, i]
        ok = (vals > 0).all() if met == "AUROC" else (vals < 0).all()
        print(f"{m}-M2 {met}: {np.round(vals, 4)} mean {vals.mean():+.4f} "
              f"consistent={'YES' if ok else 'NO'}", flush=True)
print("DONE")
