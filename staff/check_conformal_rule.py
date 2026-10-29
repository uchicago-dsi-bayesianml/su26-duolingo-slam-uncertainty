#!/usr/bin/env python3
"""Course-staff check: what does the authors' abstention threshold actually guarantee?

The authors choose the abstention threshold in analysis/stack_conformal.py (lines 43-65)
and analysis/ladder_track.py (lines 173-185): on a calibration half of the dev learners,
scan 96 quantiles of the predicted error probability and keep the largest threshold t
whose accepted tokens satisfy (errors + 1) / (accepted + 1) <= alpha. The tokens with
p <= t are "auto-handled", the rest are routed to a human. The write-up calls this
split-conformal selective classification with a finite-sample guarantee.

The SLAM data are not in the repository, so this script cannot rerun the authors'
numbers. Instead it applies their rule, line for line, to simulated data in which the
model is calibrated but, like any real model, does not know every learner's propensity
exactly. The simulated dev set is split by learner, as the authors split it, and the
script reports two things for the threshold each rule picks: how often its true error
rate among auto-handled tokens (computed from a large independent sample) is above the
target, which is what a guarantee is about, and the spread of the error rate realized on
the test half, which is what the authors compare with the target. For comparison it runs two bounded
versions of the same scan, both stopping at the first threshold that fails (the
fixed-sequence form of Learn-then-Test): a one-sided 95 percent Wilson bound that treats
tokens as independent, and a 95 percent normal bound whose variance is computed over
learners, the unit the split is made on.

The default sizes follow the es_en dev set as the authors describe it (2,640 learners,
about 288,864 tokens, docs/model_ladder_comparison.md); --small runs a set the size of
the fr_en dev set (137,564 tokens in the committed notebook output) with 1,200 learners.
The learner and token spreads are simulation choices, printed with the results.

Needs only NumPy, and takes about two minutes:
    python staff/check_conformal_rule.py [--reps N] [--small] [--seed S] [--sd-hidden SD]
"""
import argparse
from statistics import NormalDist

import numpy as np

Z95 = NormalDist().inv_cdf(0.95)
QUANTILES = np.linspace(0.05, 1.0, 96)  # the authors' grid of candidate thresholds


GH_X, GH_W = np.polynomial.hermite.hermgauss(24)


def simulate(rng, n_learners, mean_tokens, b0, sd_learner, sd_token, sd_hidden):
    """Calibrated predictions with a learner effect the model cannot see.

    The true logit is b0 + u + e + v, where u (learner) and e (token) are known to the
    model and v is a learner effect it does not know. The model's p is the exact
    probability given what it knows, E_v[sigmoid(b0 + u + e + v)], so p is calibrated,
    but tokens of one learner share v and their errors are correlated.
    """
    sizes = rng.poisson(rng.gamma(2.0, mean_tokens / 2.0, n_learners)) + 1
    learner = np.repeat(np.arange(n_learners), sizes)
    known = b0 + rng.normal(0, sd_learner, n_learners)[learner] + rng.normal(0, sd_token, learner.size)
    hidden = rng.normal(0, sd_hidden, n_learners)[learner]
    p_true = 1 / (1 + np.exp(-(known + hidden)))
    y = (rng.random(p_true.size) < p_true).astype(np.int64)
    # Gauss-Hermite quadrature over the hidden effect
    nodes = np.sqrt(2.0) * sd_hidden * GH_X
    p = np.zeros_like(known)
    for x, w in zip(nodes, GH_W / np.sqrt(np.pi)):
        p += w / (1 + np.exp(-(known + x)))
    return p, y, learner


def auroc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size)
    ranks[order] = np.arange(1, s.size + 1)
    n1 = y.sum()
    n0 = y.size - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def authors_rule(pc, yc, alpha):
    """analysis/stack_conformal.py, conformal_report(): keep the LARGEST passing threshold."""
    best = None
    for t in np.quantile(pc, QUANTILES):
        acc = pc <= t
        n = acc.sum()
        if n == 0:
            continue
        if (yc[acc].sum() + 1) / (n + 1) <= alpha:
            best = t
    return best


def wilson_upper(e, n):
    r = e / n
    centre = r + Z95**2 / (2 * n)
    half = Z95 * np.sqrt(r * (1 - r) / n + Z95**2 / (4 * n**2))
    return (centre + half) / (1 + Z95**2 / n)


def bounded_rule(pc, yc, lc, alpha, clustered):
    """Fixed-sequence scan from the smallest threshold up; stop at the first failure."""
    best = None
    n_l = lc.max() + 1
    for t in np.quantile(pc, QUANTILES):
        acc = pc <= t
        n = acc.sum()
        if n == 0:
            continue
        e = yc[acc].sum()
        if clustered:
            nl = np.bincount(lc[acc], minlength=n_l).astype(float)
            el = np.bincount(lc[acc], weights=yc[acc], minlength=n_l)
            keep = np.bincount(lc, minlength=n_l) > 0
            nl, el = nl[keep], el[keep]
            r = e / n
            m = keep.sum()
            var = m / (m - 1) * ((el - r * nl) ** 2).sum() / n**2
            ub = r + Z95 * np.sqrt(var)
        else:
            ub = wilson_upper(e, n)
        if ub <= alpha:
            best = t
        else:
            break
    return best


def population_risk(args):
    """True error rate among accepted tokens, R(t) = E[y | p <= t], from a large fresh sample.

    Because p is calibrated, E[y | p] = p, so R(t) is the mean of p over tokens with p <= t.
    """
    rng = np.random.default_rng(args.seed + 12345)
    p, _, _ = simulate(rng, 20_000, 100.0, args.b0, args.sd_learner, args.sd_token, args.sd_hidden)
    p = np.sort(p)
    csum = np.cumsum(p)

    def risk(t):
        k = np.searchsorted(p, t, side="right")
        return csum[k - 1] / k if k else 0.0

    return risk


def run(args, n_learners, mean_tokens, label, risk_of):
    rng = np.random.default_rng(args.seed)
    rules = ("authors' (errors+1)/(n+1) rule", "Wilson bound, tokens independent", "normal bound over learners")
    res = {(a, r): [] for a in args.alphas for r in rules}
    first = True
    for _ in range(args.reps):
        p, y, learner = simulate(rng, n_learners, mean_tokens, args.b0, args.sd_learner, args.sd_token, args.sd_hidden)
        if first:
            print(f"\n{label}: {n_learners:,} learners, {p.size:,} tokens in the first replicate, "
                  f"error rate {y.mean():.1%}, AUROC of the calibrated model {auroc(y, p):.3f}")
            first = False
        cal_learners = rng.choice(n_learners, size=n_learners // 2, replace=False)
        is_cal = np.zeros(n_learners, dtype=bool)
        is_cal[cal_learners] = True
        cal = is_cal[learner]
        tst = ~cal
        # relabel calibration learners 0..m-1 for the clustered bound
        _, lc = np.unique(learner[cal], return_inverse=True)
        pc, yc, pt, yt = p[cal], y[cal], p[tst], y[tst]
        for a in args.alphas:
            for r in rules:
                if r == rules[0]:
                    t = authors_rule(pc, yc, a)
                else:
                    t = bounded_rule(pc, yc, lc, a, clustered=(r == rules[2]))
                if t is None:
                    res[(a, r)].append((np.nan, np.nan, 0.0))
                    continue
                acc = pt <= t
                res[(a, r)].append((risk_of(t), yt[acc].mean() if acc.any() else 0.0, acc.mean()))
    for a in args.alphas:
        print(f"  target {a:.0%} error rate among auto-handled tokens, {args.reps} replicates:")
        for r in rules:
            arr = np.array(res[(a, r)])
            true_risk, test_risk, cov = arr[:, 0], arr[:, 1], arr[:, 2]
            ok = ~np.isnan(true_risk)
            print(f"    {r:34s} coverage {cov.mean():5.1%}; true risk of the chosen threshold above target in "
                  f"{(true_risk[ok] > a).mean():5.1%} of replicates (max {np.nanmax(true_risk):5.2%}); "
                  f"test-half risk 95th percentile {np.nanpercentile(test_risk, 95):5.2%}, max {np.nanmax(test_risk):5.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200, help="simulated dev sets (default 200)")
    ap.add_argument("--small", action="store_true", help="also run a set the size of the fr_en dev set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b0", type=float, default=-2.35, help="intercept on the logit scale")
    ap.add_argument("--sd-learner", type=float, default=0.8, help="spread of learner effects (logit scale)")
    ap.add_argument("--sd-token", type=float, default=1.3, help="spread of token effects (logit scale)")
    ap.add_argument("--sd-hidden", type=float, default=0.5, help="spread of the learner effect the model cannot see")
    args = ap.parse_args()
    args.alphas = (0.05, 0.10)
    print("Simulation choices: true logit = b0 + known learner effect + token effect + hidden learner effect, "
          f"b0 = {args.b0}, sds {args.sd_learner}, {args.sd_token} and {args.sd_hidden}. The model's p is exact "
          "given what it knows, so it is calibrated and any excess over the target comes from the rule.")
    risk_of = population_risk(args)
    run(args, 2640, 288_864 / 2640, "es_en-sized dev set", risk_of)
    if args.small:
        run(args, 1200, 137_564 / 1200, "fr_en-sized dev set", risk_of)


if __name__ == "__main__":
    main()
