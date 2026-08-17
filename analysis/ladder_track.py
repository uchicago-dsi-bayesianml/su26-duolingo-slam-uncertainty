"""Generalized model ladder for any track: python3 ladder_track.py <track>

Builds sequence features if missing, fits M1-M7 with caching
(slam/ladder_preds_<track>.npz), then stack + conformal.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from fit_models import build_vocab, shared_matrix, learner_matrix

TRACK = sys.argv[1]
tr = pd.read_parquet(f"slam/{TRACK}_train.parquet")
dv = pd.read_parquet(f"slam/{TRACK}_dev.parquet")

# ---------- sequence features ----------
def build_seq(tr, dv, path_tr, path_dv):
    tr2, dv2 = tr.copy(), dv.copy()
    tr2["split"], dv2["split"] = 0, 1
    allr = pd.concat([tr2, dv2], ignore_index=True)
    allr["row"] = np.arange(len(allr))
    allr = allr.sort_values(["user", "split", "row"], kind="stable").reset_index(drop=True)
    LAM, PRI, W = 0.99, float(tr["label"].mean()), 5.0
    out = np.zeros((len(allr), 7), dtype=np.float32)
    users, toks = allr["user"].values, allr["token"].values
    days, labs = allr["days"].values, allr["label"].values
    i, n = 0, len(allr)
    while i < n:
        j = i
        u = users[i]
        while j < n and users[j] == u:
            j += 1
        E = N = ce = cn = 0.0
        hist = {}
        for k in range(i, j):
            out[k, 0] = (E + PRI * W) / (N + W)
            out[k, 1] = (ce + PRI * W) / (cn + W)
            h = hist.get(toks[k])
            if h is None:
                out[k, 2:6] = [0.0, PRI, -1.0, 1.0]
            else:
                out[k, 2] = np.log1p(h[0])
                out[k, 3] = (h[1] + PRI * W) / (h[0] + W)
                out[k, 4] = max(days[k] - h[2], 0.0)
                out[k, 5] = 0.0
            out[k, 6] = np.log1p(k - i)
            y = labs[k]
            E = LAM * E + y; N = LAM * N + 1.0; ce += y; cn += 1.0
            if h is None:
                hist[toks[k]] = [1.0, float(y), days[k]]
            else:
                h[0] += 1.0; h[1] += float(y); h[2] = days[k]
        i = j
    cols = ["dyn_ab", "cum_err", "seen", "tok_err", "gap_days", "first", "pos"]
    feat = pd.DataFrame(out, columns=cols)
    feat["row"] = allr["row"].values
    feat = feat.sort_values("row").reset_index(drop=True)
    feat.iloc[:len(tr)].reset_index(drop=True).to_parquet(path_tr)
    feat.iloc[len(tr):].reset_index(drop=True).to_parquet(path_dv)

ptr, pdv = f"slam/{TRACK}_train_seqfeat.parquet", f"slam/{TRACK}_dev_seqfeat.parquet"
if not os.path.exists(pdv):
    build_seq(tr, dv, ptr, pdv)
    print("seq features built", flush=True)
ftr, fdv = pd.read_parquet(ptr), pd.read_parquet(pdv)

vocab = build_vocab([tr, dv])
ytr, ydv = tr["label"].values, dv["label"].values

def ece(y, p, bins=15):
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).sum())

def metrics(name, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print(f"{TRACK} {name:8s} AUROC {roc_auc_score(ydv, p):.4f}  "
          f"Brier {brier_score_loss(ydv, p):.4f}  NLL {log_loss(ydv, p):.4f}  "
          f"ECE {ece(ydv, p):.4f}", flush=True)

Xs_tr, Xs_dv = shared_matrix(tr, vocab), shared_matrix(dv, vocab)
users = sorted(tr["user"].unique())
Lu_tr, Lu_dv = learner_matrix(tr, users), learner_matrix(dv, users)
wc = tr["token"].value_counts()
wmap = {w: i for i, w in enumerate(sorted(wc[wc >= 20].index))}
def word_block(df):
    col = df["token"].map(wmap)
    ok = col.notna().values
    rows = np.where(ok)[0]
    return sp.csr_matrix((np.ones(ok.sum()), (rows, col[ok].astype(int).values)),
                         shape=(len(df), len(wmap)))
W_tr, W_dv = word_block(tr), word_block(dv)
def seq_block(f, cols):
    g = np.log1p(np.clip(f["gap_days"].values, 0, None))
    never = (f["gap_days"].values < 0).astype(float)
    m = {"dyn_ab": f["dyn_ab"].values, "cum_err": f["cum_err"].values,
         "pos": f["pos"].values, "seen": f["seen"].values,
         "tok_err": f["tok_err"].values, "gap": g, "never": never,
         "first": f["first"].values}
    return sp.csr_matrix(np.column_stack([m[c] for c in cols]))
DYN, MEM = ["dyn_ab", "cum_err", "pos"], ["seen", "tok_err", "gap", "never", "first"]
fmt_tr = tr["format"].map(vocab["format"]).values
fmt_dv = dv["format"].map(vocab["format"]).values
umap = {u: i for i, u in enumerate(users)}
def slope_block(df, fi):
    ui = df["user"].map(umap)
    ok = ui.notna().values
    rows = np.where(ok)[0]
    cols = (ui[ok].astype(int).values * 3 + fi[ok]).astype(int)
    return sp.csr_matrix((0.5 * np.ones(ok.sum()), (rows, cols)),
                         shape=(len(df), len(users) * 3))
S_tr, S_dv = slope_block(tr, fmt_tr), slope_block(dv, fmt_dv)

CACHE = f"slam/ladder_preds_{TRACK}.npz"
preds = {"label": ydv, "user": dv["user"].values}
if os.path.exists(CACHE):
    old = np.load(CACHE, allow_pickle=True)
    preds.update({k: old[k] for k in old.files})

specs = {
    "M1_pool": ([Xs_tr], [Xs_dv]),
    "M2_hier": ([Xs_tr, Lu_tr], [Xs_dv, Lu_dv]),
    "M3_irt": ([Xs_tr, Lu_tr, W_tr], [Xs_dv, Lu_dv, W_dv]),
    "M4_dyn": ([Xs_tr, Lu_tr, W_tr, seq_block(ftr, DYN)],
               [Xs_dv, Lu_dv, W_dv, seq_block(fdv, DYN)]),
    "M5_mem": ([Xs_tr, Lu_tr, W_tr, seq_block(ftr, DYN + MEM)],
               [Xs_dv, Lu_dv, W_dv, seq_block(fdv, DYN + MEM)]),
    "M6_slope": ([Xs_tr, Lu_tr, W_tr, seq_block(ftr, DYN + MEM), S_tr],
                 [Xs_dv, Lu_dv, W_dv, seq_block(fdv, DYN + MEM), S_dv]),
}
for name, (bt, bd) in specs.items():
    if name in preds:
        print(TRACK, name, "cached", flush=True); continue
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=800, tol=1e-4)
    clf.fit(sp.hstack(bt).tocsr(), ytr)
    preds[name] = clf.predict_proba(sp.hstack(bd).tocsr())[:, 1]
    metrics(name, preds[name])
    np.savez(CACHE, **preds)

if "M7_gbm" not in preds:
    def gbm_frame(df, f):
        g = pd.DataFrame({c: df[c].astype("category").cat.codes
                          for c in ["client", "session", "format", "pos", "dep", "country"]})
        g["days"], g["time"] = df["days"].values, df["time"].values
        for c in ["dyn_ab", "cum_err", "pos", "seen", "tok_err", "gap_days", "first"]:
            g["f_" + c] = f[c].values
        return g
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                        max_leaf_nodes=63, random_state=0,
                                        categorical_features=[0, 1, 2, 3, 4, 5])
    gb.fit(gbm_frame(tr, ftr), ytr)
    preds["M7_gbm"] = gb.predict_proba(gbm_frame(dv, fdv))[:, 1]
    metrics("M7_gbm", preds["M7_gbm"])
    np.savez(CACHE, **preds)

# ---------- stack + conformal ----------
names = [k for k in preds if k.startswith("M")]
P = {k: np.clip(preds[k], 1e-6, 1 - 1e-6) for k in names}
rng = np.random.default_rng(99)
uniq = np.unique(preds["user"])
cal_users = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
cal = np.array([u in cal_users for u in preds["user"]])
tst = ~cal
Z = np.column_stack([np.log(P[k] / (1 - P[k])) for k in sorted(names)])
st = LogisticRegression(C=1.0, max_iter=1000).fit(Z[cal], ydv[cal])
p_stack = st.predict_proba(Z)[:, 1]
best = max(names, key=lambda k: roc_auc_score(ydv[tst], P[k][tst]))
print(f"{TRACK} test-half AUROC: best single {best} "
      f"{roc_auc_score(ydv[tst], P[best][tst]):.4f}  stack {roc_auc_score(ydv[tst], p_stack[tst]):.4f}", flush=True)
for a in (0.05, 0.10):
    pc, yc = p_stack[cal], ydv[cal]
    bestt = None
    for t in np.quantile(pc, np.linspace(0.05, 1.0, 96)):
        acc = pc <= t
        if acc.sum() and (yc[acc].sum() + 1) / (acc.sum() + 1) <= a:
            bestt = t
    if bestt is None:
        print(f"{TRACK} conformal {a:.0%}: unattainable", flush=True); continue
    acc = p_stack[tst] <= bestt
    print(f"{TRACK} conformal stack {a:.0%}: coverage {acc.mean():.1%} "
          f"test risk {ydv[tst][acc].mean():.2%}", flush=True)
np.savez(f"slam/stack_preds_{TRACK}.npz", p_stack=p_stack, cal=cal, tst=tst)
print(TRACK, "DONE", flush=True)
