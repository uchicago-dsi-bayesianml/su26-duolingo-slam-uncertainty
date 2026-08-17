"""Leakage-safe sequence features for es_en (train+dev in chronological order).

For each token position, features use ONLY earlier tokens of that learner:
  dyn_ab   : decayed past error rate (lambda=0.99/token, prior 0.15 w=5)  [dynamic ability]
  cum_err  : undecayed past error rate (prior 0.15 w=5)
  seen     : log1p(# prior exposures of this word by this learner)         [memory/HLR]
  tok_err  : past error rate on this word (prior = dyn prior)
  gap_days : days since last exposure to this word (-1 if never seen)
  first    : 1 if never seen before
  pos      : log1p(token index within learner)
"""
import numpy as np
import pandas as pd

tr = pd.read_parquet("slam/es_en_train.parquet")
dv = pd.read_parquet("slam/es_en_dev.parquet")
tr["split"], dv["split"] = 0, 1
allr = pd.concat([tr, dv], ignore_index=True)
allr["row"] = np.arange(len(allr))
allr = allr.sort_values(["user", "split", "row"], kind="stable").reset_index(drop=True)

LAM, PRI, W = 0.99, 0.15, 5.0
out = np.zeros((len(allr), 7), dtype=np.float32)
users = allr["user"].values
toks = allr["token"].values
days = allr["days"].values
labs = allr["label"].values

i = 0
n = len(allr)
while i < n:
    j = i
    u = users[i]
    while j < n and users[j] == u:
        j += 1
    E = N = 0.0
    ce = cn = 0.0
    hist = {}  # token -> [count, errors, last_day]
    for k in range(i, j):
        out[k, 0] = (E + PRI * W) / (N + W)
        out[k, 1] = (ce + PRI * W) / (cn + W)
        t = toks[k]
        h = hist.get(t)
        if h is None:
            out[k, 2] = 0.0
            out[k, 3] = PRI
            out[k, 4] = -1.0
            out[k, 5] = 1.0
        else:
            out[k, 2] = np.log1p(h[0])
            out[k, 3] = (h[1] + PRI * W) / (h[0] + W)
            out[k, 4] = max(days[k] - h[2], 0.0)
            out[k, 5] = 0.0
        out[k, 6] = np.log1p(k - i)
        y = labs[k]
        E = LAM * E + y
        N = LAM * N + 1.0
        ce += y
        cn += 1.0
        if h is None:
            hist[t] = [1.0, float(y), days[k]]
        else:
            h[0] += 1.0
            h[1] += float(y)
            h[2] = days[k]
    i = j

cols = ["dyn_ab", "cum_err", "seen", "tok_err", "gap_days", "first", "pos"]
feat = pd.DataFrame(out, columns=cols)
feat["row"] = allr["row"].values
feat = feat.sort_values("row").reset_index(drop=True)
ftr = feat.iloc[:len(tr)].reset_index(drop=True)
fdv = feat.iloc[len(tr):].reset_index(drop=True)
ftr.to_parquet("slam/es_en_train_seqfeat.parquet")
fdv.to_parquet("slam/es_en_dev_seqfeat.parquet")
print("train feat", ftr.shape, "dev feat", fdv.shape)
print(fdv[cols].describe().round(3).to_string())
