"""Deck figures: (1) transfer-matrix heatmap, (2) risk-coverage with conformal
operating points, (3) model-ladder dot plot across tracks. Static PNGs, light
surface, reference palette."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2, MUT, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "text.color": INK, "axes.edgecolor": BASE, "xtick.color": MUT,
                     "ytick.color": MUT, "axes.labelcolor": INK2, "figure.dpi": 200})

# ---------------- Fig 1: transfer matrix ----------------
M = np.array([[0.7765, 0.7322, 0.7733],
              [0.7518, 0.7554, 0.7567],
              [0.7662, 0.7293, 0.7769]])
tracks = ["fr_en", "es_en", "en_es"]
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
fig, ax = plt.subplots(figsize=(5.4, 4.4), facecolor=SURF)
ax.set_facecolor(SURF)
vmin, vmax = 0.725, 0.782
ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
for i in range(3):
    for j in range(3):
        v = M[i, j]
        frac = (v - vmin) / (vmax - vmin)
        color = "#ffffff" if frac > 0.55 else INK
        w = "bold" if i == j else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=13,
                color=color, fontweight=w)
# 2px surface gaps between cells
for k in [0.5, 1.5]:
    ax.axhline(k, color=SURF, lw=2)
    ax.axvline(k, color=SURF, lw=2)
ax.set_xticks(range(3), tracks, fontsize=11)
ax.set_yticks(range(3), tracks, fontsize=11)
ax.set_xlabel("evaluated on (dev set)", fontsize=11)
ax.set_ylabel("fit on (train set)", fontsize=11)
ax.set_title("Cross-track transfer: AUROC of the hierarchical model",
             fontsize=12.5, color=INK, loc="left", pad=40)
ax.text(0, 1.015, "Diagonal = native fit (bold). Off-diagonal drop is\nreproducible: clustered 95% CI on transfer ±0.005.",
        transform=ax.transAxes, fontsize=8.6, color=INK2, va="bottom")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("fig1_transfer_matrix.png", facecolor=SURF, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: risk-coverage ----------------
d = np.load("slam/ladder_preds.npz", allow_pickle=True)
s = np.load("slam/stack_preds.npz")
y = d["label"].astype(int)
tst = s["tst"]
curves = {"Final GLM (M6)": (np.clip(d["M6_slope"], 1e-6, 1), BLUE),
          "Stacked ensemble": (np.clip(s["p_stack"], 1e-6, 1), ORANGE)}
fig, ax = plt.subplots(figsize=(6.2, 4.3), facecolor=SURF)
ax.set_facecolor(SURF)
for name, (p, c) in curves.items():
    pt, yt = p[tst], y[tst]
    o = np.argsort(pt)
    ys = yt[o]
    cov = np.arange(1, len(ys) + 1) / len(ys)
    risk = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    sl = slice(200, None, 200)
    ax.plot(cov[sl] * 100, risk[sl] * 100, color=c, lw=2, label=name)
# conformal operating points (stack): 5% and 10% budgets
for covp, riskp, lbl in [(50.5, 5.03, "5% budget:\n50% auto-handled"),
                         (82.9, 9.79, "10% budget:\n83% auto-handled")]:
    ax.scatter([covp], [riskp], s=64, color=ORANGE, zorder=5,
               edgecolor=SURF, linewidth=2)
    ax.annotate(lbl, (covp, riskp), textcoords="offset points", xytext=(-8, 14),
                fontsize=8.8, color=INK2, ha="right")
ax.axhline(15.7, color=BASE, lw=1, ls=(0, (4, 3)))
ax.text(98, 14.9, "base error rate 15.7%", fontsize=8.6, color=MUT, ha="right")
ax.set_xlabel("coverage — share of tokens auto-handled (%)", fontsize=10.5)
ax.set_ylabel("error rate among auto-handled (%)", fontsize=10.5)
ax.set_title("Selective prediction: risk vs coverage (es_en dev, held-out learners)",
             fontsize=12.5, color=INK, loc="left", pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(0, 100)
ax.set_ylim(0, 18)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("fig2_risk_coverage.png", facecolor=SURF, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: model ladder dot plot ----------------
models = ["Pooled", "+ learner\n(current)", "+ item difficulty\n(IRT)",
          "+ dynamic\nability", "+ memory\n(HLR)", "+ learner×format\nslopes", "Stack"]
vals = {"es_en": [0.6700, 0.7144, 0.7554, 0.7611, 0.7709, 0.7800, 0.7937],
        "fr_en": [0.7056, 0.7317, 0.7768, 0.7802, 0.7917, 0.8001, 0.8131],
        "en_es": [0.6969, 0.7372, 0.7797, 0.7863, 0.7947, 0.8061, 0.8119]}
cols = {"es_en": BLUE, "fr_en": ORANGE, "en_es": AQUA}
fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor=SURF)
ax.set_facecolor(SURF)
ypos = np.arange(len(models))[::-1]
for t in tracks_order if (tracks_order := ["es_en", "fr_en", "en_es"]) else []:
    pass
for t in ["es_en", "fr_en", "en_es"]:
    ax.plot(vals[t], ypos, color=cols[t], lw=1.2, alpha=0.45, zorder=2)
    ax.scatter(vals[t], ypos, s=52, color=cols[t], zorder=4,
               edgecolor=SURF, linewidth=1.5, label=t)
# direct labels on last point (relief for aqua + identity)
ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor="#52514e",
          bbox_to_anchor=(0.02, 0.02))
ax.set_yticks(ypos, models, fontsize=9.5)
ax.set_xlabel("AUROC (official dev set, native fit per track)", fontsize=10.5)
ax.set_title("Model ladder: item difficulty is the dominant gain on every track",
             fontsize=12.5, color=INK, loc="left", pad=10)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(0.66, 0.825)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig("fig3_model_ladder.png", facecolor=SURF, bbox_inches="tight")
plt.close(fig)
print("figures written")
