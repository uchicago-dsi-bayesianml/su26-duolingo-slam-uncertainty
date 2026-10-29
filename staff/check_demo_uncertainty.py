#!/usr/bin/env python3
"""Course-staff check: what the demo's posterior draws say about the uncertainty claims.

demo_data.json is the only model output in the repository: 300 dev tokens from the
French course (the course the model was fit on) and 300 from the Spanish course (never
seen), each with the posterior-mean error probability p, an aleatoric and an epistemic
value, the true label and 120 posterior-predictive draws of p. The meta note says the
draws come from the hierarchical ADVI posterior fit on fr_en. The code that produced the
file is not in the repository.

The script checks four things:
  1. that p is the mean of the draws, that "aleatoric" is the expected binary entropy of
     the draws and "epistemic" the mutual information H(p) - aleatoric, in nats;
  2. how large the epistemic part is next to the aleatoric part;
  3. the deck's claim that epistemic uncertainty "rises 4.6x under cross-course shift",
     using only the 60 + 60 tokens in the file's "random" category, the one subset that
     is not selected for being extreme, with a bootstrap interval;
  4. what the demo's own routing rule (demo.html, render()) reads: it defers when
     |p - 0.5| is below the slider value and never looks at the epistemic value.

The tokens are a curated sample (60 per category per course), so nothing here estimates
dev-set coverage or risk; staff/check_conformal_rule.py covers the threshold rule.

Needs only NumPy:  python staff/check_demo_uncertainty.py [--demo PATH] [--boot B]
"""
import argparse
import json
import pathlib

import numpy as np


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default=str(repo / "demo_data.json"))
    ap.add_argument("--boot", type=int, default=10_000, help="bootstrap resamples (default 10,000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.demo, encoding="utf-8"))
    print("meta:", d["meta"])
    sides = {}
    for side in ("home", "transfer"):
        rows = d[side]
        draws = np.array([r["draws"] for r in rows], dtype=float)
        sides[side] = dict(
            track=rows[0]["track"],
            cat=np.array([r["cat"] for r in rows]),
            p=np.array([r["p"] for r in rows]),
            alea=np.array([r["alea"] for r in rows]),
            epi=np.array([r["epi"] for r in rows]),
            label=np.array([r["label"] for r in rows]),
            draws=draws,
        )

    print("\n1. Definitions (max absolute difference over the tokens; the file rounds to 3-4 decimals")
    print("   and keeps 120 of the draws, so small differences are expected):")
    for side, s in sides.items():
        mean_draw = s["draws"].mean(1)
        exp_ent = entropy(s["draws"]).mean(1)
        print(f"   {s['track']}: {len(s['p'])} tokens, {s['draws'].shape[1]} draws each")
        print(f"     p vs mean of the stored draws                 {np.abs(s['p'] - mean_draw).max():.4f}")
        print(f"     aleatoric vs expected entropy of the draws    {np.abs(s['alea'] - exp_ent).max():.4f}")
        print(f"     epistemic vs H(p) - aleatoric                 {np.abs(s['epi'] - (entropy(s['p']) - s['alea'])).max():.4f}")

    print("\n2. Size of the two parts (nats); total = H(p):")
    for side, s in sides.items():
        share = s["epi"] / entropy(s["p"])
        print(f"   {s['track']}: mean aleatoric {s['alea'].mean():.3f}, mean epistemic {s['epi'].mean():.4f}, "
              f"epistemic share of the total {np.median(share):.1%} median, {share.max():.1%} max; "
              f"posterior sd of p {s['draws'].std(1).mean():.3f} on average")
    print("   by category, mean epistemic (home / transfer):")
    for c in sorted(set(sides["home"]["cat"])):
        h = sides["home"]["epi"][sides["home"]["cat"] == c].mean()
        t = sides["transfer"]["epi"][sides["transfer"]["cat"] == c].mean()
        print(f"     {c:18s} {h:.4f} / {t:.4f}")

    print("\n3. Epistemic ratio, transfer over home, on the 'random' category only:")
    rng = np.random.default_rng(args.seed)
    eh = sides["home"]["epi"][sides["home"]["cat"] == "random"]
    et = sides["transfer"]["epi"][sides["transfer"]["cat"] == "random"]
    ratio = et.mean() / eh.mean()
    boots = np.array([rng.choice(et, et.size).mean() / rng.choice(eh, eh.size).mean() for _ in range(args.boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"   {eh.size} home and {et.size} transfer tokens: mean epistemic {eh.mean():.4f} vs {et.mean():.4f}, "
          f"ratio {ratio:.2f}, bootstrap 95% interval [{lo:.2f}, {hi:.2f}]")
    allratio = sides["transfer"]["epi"].mean() / sides["home"]["epi"].mean()
    print(f"   all 300 + 300 tokens (stratified, for reference): ratio {allratio:.2f}")
    for side in ("home", "transfer"):
        s = sides[side]
        m = s["cat"] == "random"
        print(f"   'random' {side}: mean p {s['p'][m].mean():.3f}, observed error rate {s['label'][m].mean():.3f}")

    print("\n4. The demo's routing rule, defer when |p - 0.5| < slider (default 15 points):")
    for side, s in sides.items():
        defer = np.abs(s["p"] - 0.5) < 0.15
        hi_epi = s["epi"] >= np.quantile(s["epi"], 0.8)
        print(f"   {s['track']}: defers on {defer.mean():.0%} of the 300 tokens; of the fifth with the highest "
              f"epistemic value it defers on {defer[hi_epi].mean():.0%}, of the rest on {defer[~hi_epi].mean():.0%}")
        r = np.corrcoef(np.abs(s["p"] - 0.5), s["epi"])[0, 1]
        print(f"     correlation of |p - 0.5| with the epistemic value: {r:.2f}")
    print("   The rule reads only p; the epistemic bar is displayed but does not enter the decision.")


if __name__ == "__main__":
    main()
