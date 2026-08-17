"""Analysis 1: learner-clustered bootstrap CIs for transfer AUROCs on es_en dev."""
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260805)
d = np.load("slam/es_dev_preds.npz", allow_pickle=True)
y = d["label"].astype(int)
users = d["user"]
models = ["p_es_en", "p_fr_en", "p_en_es", "p_es_en_global"]
P = {m: d[m] for m in models}

uniq, inv = np.unique(users, return_inverse=True)
n_users = len(uniq)
# index lists per user
order = np.argsort(inv, kind="stable")
inv_sorted = inv[order]
starts = np.searchsorted(inv_sorted, np.arange(n_users))
ends = np.searchsorted(inv_sorted, np.arange(n_users) + 1)
user_idx = [order[s:e] for s, e in zip(starts, ends)]

B = 1000
aucs = {m: np.empty(B) for m in models}
for b in range(B):
    sel = rng.integers(0, n_users, n_users)
    idx = np.concatenate([user_idx[u] for u in sel])
    yb = y[idx]
    if yb.min() == yb.max():
        continue
    for m in models:
        aucs[m][b] = roc_auc_score(yb, P[m][idx])

def ci(a):
    return np.percentile(a, [2.5, 97.5])

print("Point AUROCs (full dev):")
for m in models:
    print(f"  {m}: {roc_auc_score(y, P[m]):.4f}  boot 95% CI [{ci(aucs[m])[0]:.4f}, {ci(aucs[m])[1]:.4f}]  SE {aucs[m].std():.4f}")

pairs = [("p_es_en", "p_fr_en"), ("p_es_en", "p_en_es"), ("p_fr_en", "p_en_es"),
         ("p_es_en_global", "p_fr_en"), ("p_es_en", "p_es_en_global")]
print("\nPaired deltas (same resamples):")
for a, bm in pairs:
    delta = aucs[a] - aucs[bm]
    lo, hi = ci(delta)
    point = roc_auc_score(y, P[a]) - roc_auc_score(y, P[bm])
    p_gt = (delta > 0).mean()
    print(f"  {a} - {bm}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0)={p_gt:.3f}")

np.savez("slam/bootstrap_aucs.npz", **aucs)
