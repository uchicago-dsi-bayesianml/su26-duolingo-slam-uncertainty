"""Analysis 4: joint multi-track hierarchical model (MAP analogue).

logit P(err) = B_global . x + gamma * B_track . x + a_learner
Track-deviation blocks are shrunk (columns scaled by gamma=0.5 == stronger
prior), mimicking partial pooling across tracks. The pairwise similarity of
effective deviation vectors (gamma * B_track) is the quantitative version of
'languages as points in a latent space'.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fit_models import build_vocab, shared_matrix, learner_matrix, CATS

GAMMA = 0.5
N_PER_TRACK = 1_000_000
rng = np.random.default_rng(11)

tracks = {t: pd.read_parquet(f"slam/{t}_train.parquet") for t in ["fr_en", "es_en", "en_es"]}
dev = pd.read_parquet("slam/es_en_dev.parquet")
vocab = build_vocab(list(tracks.values()) + [dev])
names = ["fr_en", "es_en", "en_es"]

parts = []
for t in names:
    df = tracks[t]
    if len(df) > N_PER_TRACK:
        df = df.sample(N_PER_TRACK, random_state=5)
    df = df.copy()
    df["track"] = t
    df["user"] = t + ":" + df["user"]  # track-scoped learner ids
    parts.append(df)
alld = pd.concat(parts, ignore_index=True)
print("joint rows:", len(alld), flush=True)

Xs = shared_matrix(alld, vocab)
n_shared = Xs.shape[1]
dev_blocks = []
for t in names:
    mask = (alld["track"] == t).values.astype(float)
    D = sp.diags(mask) @ Xs * GAMMA
    dev_blocks.append(D.tocsr())
users = sorted(alld["user"].unique())
X = sp.hstack([Xs] + dev_blocks + [learner_matrix(alld, users)]).tocsr()
y = alld["label"].values
print("design:", X.shape, "nnz:", X.nnz, flush=True)

clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
clf.fit(X, y)
w = clf.coef_.ravel()
Wg = w[:n_shared]
Wd = {t: GAMMA * w[n_shared * (i + 1):n_shared * (i + 2)] for i, t in enumerate(names)}
np.savez("slam/joint_model.npz", global_=Wg, **{f"dev_{t}": Wd[t] for t in names})

# ---- similarity of deviation vectors ----
# group index ranges within the shared block
ranges, off = {}, 0
for c in CATS:
    ranges[c] = (off, off + len(vocab[c])); off += len(vocab[c])
ranges["morph"] = (off, off + len(vocab["_morph"])); off += len(vocab["_morph"])
ranges["numeric"] = (off, off + 3)

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan

pairs = [("fr_en", "es_en"), ("fr_en", "en_es"), ("es_en", "en_es")]
print("\n=== Cosine similarity of track deviation vectors ===")
print("overall:", {f"{a}~{b}": round(cos(Wd[a], Wd[b]), 3) for a, b in pairs})
LING = ["pos", "dep", "morph"]
for g in LING + ["format", "session", "numeric"]:
    s, e = ranges[g]
    print(f"  {g:8s}", {f"{a}~{b}": round(cos(Wd[a][s:e], Wd[b][s:e]), 3) for a, b in pairs})
ling_idx = np.concatenate([np.arange(*ranges[g]) for g in LING])
print("linguistic (pos+dep+morph):",
      {f"{a}~{b}": round(cos(Wd[a][ling_idx], Wd[b][ling_idx]), 3) for a, b in pairs})
print("deviation norms:", {t: round(float(np.linalg.norm(Wd[t])), 3) for t in names})

# ---- predictive check on es_en dev (no learner effects) ----
Xd = shared_matrix(dev, vocab)
b0 = clf.intercept_[0]
yd = dev["label"].values
print("\n=== es_en dev AUROC (joint model, no learner ints) ===")
for lbl, vec in [("global-only", None), ("global+es dev-block", "es_en"),
                 ("global+fr dev-block", "fr_en"), ("global+en_es dev-block", "en_es")]:
    z = Xd @ Wg + b0
    if vec:
        z = z + Xd @ Wd[vec]
    print(f"  {lbl}: {roc_auc_score(yd, z):.4f}")
