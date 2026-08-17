"""Analysis 2: fit-side variance — refit fr_en (and es_en) on learner-bootstrap
resamples of the training set; evaluate transfer AUROC on the FIXED es_en dev.
lbfgs is deterministic, so all variance comes from training-population resampling."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_models import build_vocab, fit_track, predict

rng = np.random.default_rng(7)
tracks = {t: pd.read_parquet(f"slam/{t}_train.parquet") for t in ["fr_en", "es_en", "en_es"]}
dev = pd.read_parquet("slam/es_en_dev.parquet")
vocab = build_vocab(list(tracks.values()) + [dev])
y = dev["label"].values

def learner_bootstrap(df, rng):
    users = df["user"].unique()
    sel = rng.choice(users, size=len(users), replace=True)
    parts = []
    groups = dict(tuple(df.groupby("user")))
    for k, u in enumerate(sel):
        g = groups[u].copy()
        g["user"] = f"{u}__{k}"  # duplicated learners become distinct levels
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

results = {"fr_en": [], "es_en": []}
K_FR, K_ES = 10, 5
for k in range(K_FR):
    boot = learner_bootstrap(tracks["fr_en"], rng)
    clf, users, ns = fit_track(boot, vocab)
    auc = roc_auc_score(y, predict(clf, users, ns, dev, vocab, use_learner=False))
    results["fr_en"].append(auc)
    print(f"fr_en refit {k}: transfer AUROC {auc:.4f}", flush=True)
for k in range(K_ES):
    boot = learner_bootstrap(tracks["es_en"], rng)
    clf, users, ns = fit_track(boot, vocab)
    # global-only eval (fair comparison to transfer: no learner intercepts)
    auc = roc_auc_score(y, predict(clf, users, ns, dev, vocab, use_learner=False))
    results["es_en"].append(auc)
    print(f"es_en refit {k}: global-only AUROC {auc:.4f}", flush=True)

for t, v in results.items():
    v = np.array(v)
    print(f"{t}: mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  range [{v.min():.4f}, {v.max():.4f}]")
np.savez("slam/fit_variance.npz", fr_en=np.array(results["fr_en"]), es_en=np.array(results["es_en"]))
