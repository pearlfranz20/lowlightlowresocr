"""
Renders assets/*.png from benchmark_results.json for the README. Run
benchmark.py first. Colors follow the dataviz palette convention: each
pathway keeps one color across every chart (dual=blue, parvo=orange,
magno=aqua).

Run: python plot_results.py
"""

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

BLUE = "#2a78d6"    # dual_pathway
ORANGE = "#eb6834"  # parvo_only
AQUA = "#1baf7a"    # magno_only
YELLOW = "#eda100"  # secondary condition (not a pathway)

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_severity_curve(results, out_path):
    curve = results["severity_curve"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    _clean_axes(ax)

    series = [("dual_pathway", "dual_pathway", BLUE), ("parvo_only", "parvo_only", ORANGE),
              ("magno_only", "magno_only", AQUA)]
    for key, label, color in series:
        by_x = {float(k): v for k, v in curve[key].items()}
        xs = sorted(by_x.keys())
        ys = [by_x[x] for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                 markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                 label=label, zorder=3, solid_capstyle="round")

    ax.set_xlabel("corruption severity (this benchmark's sweep, not degrade_p)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Accuracy vs. corruption severity (held-out synthetic test set)",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.annotate("dual_pathway and magno_only overlap exactly —\nthe gate collapsed onto magno (see gate-weight chart)",
                xy=(0.5, 0.888), xytext=(0.04, 0.42), fontsize=9, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gate_weights(results, out_path):
    gw = results["gate_weights"]
    conditions = ["Clean", "Heavily corrupted"]
    magno_vals = [gw["clean"][0], gw["degraded"][0]]
    parvo_vals = [gw["clean"][1], gw["degraded"][1]]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    _clean_axes(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(BASELINE)
    ax.grid(False)

    y = range(len(conditions))
    bar_h = 0.5
    ax.barh(y, magno_vals, height=bar_h, color=AQUA, label="magno weight", zorder=3)
    ax.barh(y, parvo_vals, height=bar_h, left=magno_vals, color=ORANGE, label="parvo weight", zorder=3)

    for i, (m, p) in enumerate(zip(magno_vals, parvo_vals)):
        ax.text(m / 2, i, f"{m:.0%}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if p > 0.03:
            ax.text(m + p / 2, i, f"{p:.0%}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(conditions)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("average gate weight (dual_pathway model, held-out test set)")
    ax.set_title("Learned gate weighting: clean vs. corrupted input",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_text_metrics(results, out_path):
    tm = results["text_metrics"]
    metrics = ["iou", "precision", "recall", "f1"]
    labels = ["IoU", "Precision", "Recall", "F1"]
    clean_vals = [tm["clean"][m] for m in metrics]
    degraded_vals = [tm["degraded"][m] for m in metrics]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    _clean_axes(ax)

    ax.bar([i - width / 2 for i in x], clean_vals, width=width, color=BLUE, label="Clean", zorder=3)
    ax.bar([i + width / 2 for i in x], degraded_vals, width=width, color=YELLOW, label="Heavily corrupted", zorder=3)

    for i, (c, d) in enumerate(zip(clean_vals, degraded_vals)):
        ax.text(i - width / 2, c + 0.02, f"{c:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
        ax.text(i + width / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("TextRegionHead detection quality: clean vs. heavily corrupted",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    with open("benchmark_results.json") as f:
        results = json.load(f)

    plot_severity_curve(results, "assets/severity_curve.png")
    plot_gate_weights(results, "assets/gate_weights.png")
    plot_text_metrics(results, "assets/text_metrics.png")
    print("Wrote assets/severity_curve.png, assets/gate_weights.png, assets/text_metrics.png")


if __name__ == "__main__":
    main()
