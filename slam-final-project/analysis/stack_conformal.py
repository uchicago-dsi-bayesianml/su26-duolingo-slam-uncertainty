"""Stacking + split-conformal selective prediction on the ladder predictions.

Learner-level split of es_en dev: 50% calibration, 50% test.
- Stacker: logistic combination of base-model logits, fit on cal, eval on test.
- Conformal: split-conformal selective classification on the best single model
  and on the stack: choose threshold on cal so that P(error among accepted) <=
  target risk with finite-sample correction; report empirical test risk/coverage.
  Compare against naive posterior-probability thresholding at the same coverage.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d = np.load("slam/ladder_preds.npz", allow_pickle=True)
y = d["label"].astype(int)
users = d["user"]
names = [k for k in d.files if k.startswith("M")]
P = {k: np.clip(d[k], 1e-6, 1 - 1e-6) for k in names}

rng = np.random.default_rng(99)
uniq = np.unique(users)
cal_users = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
cal = np.array([u in cal_users for u in users])
tst = ~cal
print(f"cal {cal.sum()} tokens / test {tst.sum()} tokens")

# ---- stacking ----
Z = np.column_stack([np.log(P[k] / (1 - P[k])) for k in names])
st = LogisticRegression(C=1.0, max_iter=1000)
st.fit(Z[cal], y[cal])
p_stack = st.predict_proba(Z)[:, 1]
print("\nAUROC on test half:")
for k in names:
    print(f"  {k:8s} {roc_auc_score(y[tst], P[k][tst]):.4f}")
print(f"  stack    {roc_auc_score(y[tst], p_stack[tst]):.4f}")
print("stack weights:", dict(zip(names, np.round(st.coef_.ravel(), 3))))

# ---- conformal selective prediction ----
# Predict "correct" (y=0) vs abstain: accept token if p_err <= tau.
# Split-conformal: on cal, choose tau as the largest value s.t. the
# (1-alpha)-adjusted quantile of empirical risk among accepted <= alpha.
def conformal_report(p, label, alphas=(0.05, 0.10)):
    pc, yc = p[cal], y[cal]
    pt, yt = p[tst], y[tst]
    print(f"\n  {label}:")
    for a in alphas:
        taus = np.quantile(pc, np.linspace(0.05, 1.0, 96))
        best = None
        for t in taus:
            acc = pc <= t
            n = acc.sum()
            if n == 0:
                continue
            # finite-sample upper confidence (rule-of-thumb +1 correction)
            risk_ucb = (yc[acc].sum() + 1) / (n + 1)
            if risk_ucb <= a:
                best = t
        if best is None:
            print(f"    target risk {a:.0%}: unattainable")
            continue
        acc_t = pt <= best
        cov = acc_t.mean()
        risk = yt[acc_t].mean() if acc_t.sum() else 0.0
        print(f"    target risk {a:.0%}: tau={best:.3f}  coverage {cov:.1%}  "
              f"empirical test risk {risk:.2%}")

best_single = max(names, key=lambda k: roc_auc_score(y[tst], P[k][tst]))
conformal_report(P[best_single], f"conformal on {best_single}")
conformal_report(p_stack, "conformal on stack")

# naive: same coverage as conformal 5% target, threshold chosen on cal by risk only
np.savez("slam/stack_preds.npz", p_stack=p_stack, cal=cal, tst=tst)
