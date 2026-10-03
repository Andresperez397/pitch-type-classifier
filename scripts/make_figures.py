"""Build report figures from reports/tables/ (run after run_analysis.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
LABELS = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c8ef", "#2a78d6", "#123a6b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def feature_set_comparison():
    """Paired per repeat: same held-out pitchers, absolute vs relative features."""
    r = pd.read_csv(TAB / "repeats.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = {"logistic": "Logistic regression", "gbm": "Gradient-boosted trees"}
    ticks, labels = [], []
    for g, kind in enumerate(["gbm", "logistic"]):
        a = r[(r.model == kind) & (r.features == "A_absolute")].set_index("seed")["macro_f1"]
        b = r[(r.model == kind) & (r.features == "B_relative")].set_index("seed")["macro_f1"]
        for seed in a.index:
            y = g * 6.5 + seed
            ax.plot([a[seed], b[seed]], [y, y], color=GRID, lw=2, zorder=1)
            ax.plot(a[seed], y, "o", ms=7, color=ORANGE, zorder=2,
                    label="Absolute features" if (g, seed) == (0, 0) else None)
            ax.plot(b[seed], y, "o", ms=7, color=BLUE, zorder=2,
                    label="Pitcher-relative features" if (g, seed) == (0, 0) else None)
        ticks.append(g * 6.5 + 2)
        labels.append(f"{names[kind]}\n(+{(b - a).mean():.3f}, wins {(b > a).sum()}/5)")
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Macro-F1 on held-out pitchers (one row per repeat)")
    ax.set_title("Relative features win in every repeat")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_feature_sets.png", dpi=200)
    plt.close(fig)


def confusion():
    cm = pd.read_csv(TAB / "confusion_seed0.csv", index_col=0).loc[LABELS, LABELS]
    rown = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(rown.to_numpy(), cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = rown.iat[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.0%}", ha="center",
                        va="center", fontsize=8, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Model label")
    ax.set_ylabel("MLB label")
    ax.grid(False)
    ax.set_title("Where the model and MLB disagree (row %, held-out pitchers)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_confusion.png", dpi=200)
    plt.close(fig)


def per_class():
    pc = pd.read_csv(TAB / "per_class_f1.csv").set_index(["features", "model"])
    a = pc.loc[("A_absolute", "gbm"), LABELS]
    b = pc.loc[("B_relative", "gbm"), LABELS]
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(x - 0.2, a, width=0.38, color=ORANGE, label="Absolute")
    ax.bar(x + 0.2, b, width=0.38, color=BLUE, label="Pitcher-relative")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (held-out pitchers)")
    ax.set_title("Per-pitch-type accuracy, gradient-boosted trees")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_per_class.png", dpi=200)
    plt.close(fig)


def error_kinds():
    p = pd.read_csv(TAB / "error_pairs_seed0.csv").head(8)
    lab = [f"{t} → {q}" for t, q in zip(p["true"], p["pred"])]
    y = np.arange(len(p))[::-1]
    amb = p["ambiguous"].to_numpy()
    me = p["model_error"].to_numpy()
    mixed = 1 - amb - me
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh(y, amb, color=BLUE, label="Neighbors agree with model (label ambiguity)")
    ax.barh(y, mixed, left=amb, color="#c3c2b7", label="Mixed", edgecolor=SURFACE, linewidth=2)
    ax.barh(y, me, left=amb + mixed, color=ORANGE, label="Neighbors agree with MLB (model error)",
            edgecolor=SURFACE, linewidth=2)
    for yi, n in zip(y, p["n"]):
        ax.text(1.01, yi, f"{n:,}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, lab)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share of misclassified pitches (n at right)")
    ax.set_title("The 8 most common disagreements: model error or label ambiguity?")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_error_kinds.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    feature_set_comparison()
    confusion()
    per_class()
    error_kinds()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
