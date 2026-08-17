# === Drop-in Colab cell: learner-clustered bootstrap for the transfer gap ===
# Tests whether AUROC differences on es_en dev exceed learner-level sampling
# noise, using YOUR fitted models' predictions (posterior means from saved draws).
#
# Assumes (rename to match the CrossTrack notebook):
#   es_dev_df    : es_en dev DataFrame with columns 'user' and 'label',
#                  rows in the SAME order as the prediction arrays
#   draws_native : np.load('shane_outputs/es_en_dev_hier_draws.npy')      # es_en-fit
#   draws_fr     : np.load('shane_outputs/es_en_dev_hier_draws_fitfr.npy') # fr_en-fit
# Draws are (n_draws, n_tokens) probabilities; posterior mean = axis-0 mean.

import numpy as np
from sklearn.metrics import roc_auc_score

p_native = draws_native.mean(axis=0)
p_fr = draws_fr.mean(axis=0)
y = es_dev_df["label"].to_numpy().astype(int)
users = es_dev_df["user"].to_numpy()
assert len(y) == len(p_native) == len(p_fr)

rng = np.random.default_rng(0)
uniq, inv = np.unique(users, return_inverse=True)
order = np.argsort(inv, kind="stable")
inv_sorted = inv[order]
bounds = [np.searchsorted(inv_sorted, i) for i in range(len(uniq) + 1)]
user_idx = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]

B = 1000
auc_n, auc_f = np.empty(B), np.empty(B)
for b in range(B):
    sel = rng.integers(0, len(uniq), len(uniq))
    idx = np.concatenate([user_idx[u] for u in sel])
    yb = y[idx]
    auc_n[b] = roc_auc_score(yb, p_native[idx])
    auc_f[b] = roc_auc_score(yb, p_fr[idx])

delta = auc_n - auc_f
lo, hi = np.percentile(delta, [2.5, 97.5])
print(f"native  AUROC {roc_auc_score(y, p_native):.4f}  "
      f"95% CI [{np.percentile(auc_n, 2.5):.4f}, {np.percentile(auc_n, 97.5):.4f}]")
print(f"fr-fit  AUROC {roc_auc_score(y, p_fr):.4f}  "
      f"95% CI [{np.percentile(auc_f, 2.5):.4f}, {np.percentile(auc_f, 97.5):.4f}]")
print(f"paired delta (native - transfer): "
      f"{roc_auc_score(y, p_native) - roc_auc_score(y, p_fr):+.4f}  "
      f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) = {(delta > 0).mean():.3f}")
# Interpretation: if the CI excludes zero, the transfer gap is real
# (not eval-side sampling variation). Expected CI half-width with
# ~2,640 learner clusters is roughly ±0.003-0.006.
