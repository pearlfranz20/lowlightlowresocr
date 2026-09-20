"""
Renders assets/cure_tsr_*.png from cure_tsr_results.json. Run
benchmark_cure_tsr.py first (needs CURE-TSR downloaded — see its docstring).

Unlike the other severity-curve charts (synthetic corruption or GTSRB's own
sweep), the x-axis here is CURE-TSR's own 0-5 challenge-level scale, and
there's one small panel per challenge type rather than one curve — the
whole point of this benchmark is that "varied quality" isn't one axis, it's
several independent ones (darkening vs. noise vs. losing color entirely
behave differently), so collapsing them into a single line would hide that.
"""

import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_results import (
    AQUA, BASELINE, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    ORANGE, PURPLE, SURFACE, BLUE, _clean_axes,
)

MODEL_STYLE = {
    "dual_pathway": (BLUE, "dual_pathway"),
    "dual_pathway_lateral": (PURPLE, "dual_pathway_lateral"),
    "parvo_only": (ORANGE, "parvo_only"),
    "magno_only": (AQUA, "magno_only"),
}


def plot_severity_grid(results, out_path):
    curve = results["severity_curve"]
    challenges = list(curve.keys())
    ncols = min(3, len(challenges))
    nrows = math.ceil(len(challenges) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), dpi=200, squeeze=False)
    for i, challenge in enumerate(challenges):
        ax = axes[i // ncols][i % ncols]
        _clean_axes(ax)
        per_model = curve[challenge]
        for model_name, (color, label) in MODEL_STYLE.items():
            if model_name not in per_model:
                continue
            by_level = {int(k): v for k, v in per_model[model_name].items()}
            xs = sorted(by_level.keys())
            ys = [by_level[x] for x in xs]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
                     markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=0.8,
                     label=label, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(challenge, fontsize=10.5, color=INK_PRIMARY, loc="left")

    # blank any unused grid cells
    for j in range(len(challenges), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs. CURE-TSR challenge level (real photos, real degradation)",
                 fontsize=13, color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gate_weights_comparison(results, out_path):
    gw = results["gate_weights"]
    models = [m for m in ("dual_pathway", "dual_pathway_lateral") if m in gw]
    conditions = ["Clean", "Degraded (avg. across tested challenges, level 5)"]

    fig, axes = plt.subplots(len(models), 1, figsize=(7.5, 2.2 * len(models) + 1), dpi=200, squeeze=False)
    for row, model_name in enumerate(models):
        ax = axes[row][0]
        _clean_axes(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(BASELINE)
        ax.grid(False)

        magno_vals = [gw[model_name]["clean"][0], gw[model_name]["degraded"][0]]
        parvo_vals = [gw[model_name]["clean"][1], gw[model_name]["degraded"][1]]
        y = range(len(conditions))
        ax.barh(y, magno_vals, height=0.5, color=AQUA, label="magno weight", zorder=3)
        ax.barh(y, parvo_vals, height=0.5, left=magno_vals, color=ORANGE, label="parvo weight", zorder=3)
        for i, (m, p) in enumerate(zip(magno_vals, parvo_vals)):
            ax.text(m / 2, i, f"{m:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            if p > 0.03:
                ax.text(m + p / 2, i, f"{p:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_yticks(list(y))
        ax.set_yticklabels(conditions, fontsize=9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(model_name, fontsize=11, color=INK_PRIMARY, loc="left", pad=8)

    axes[0][0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=2)
    fig.suptitle("Learned gate weighting on real CURE-TSR degradation",
                 fontsize=12.5, color=INK_PRIMARY, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    with open("cure_tsr_results.json") as f:
        results = json.load(f)
    plot_severity_grid(results, "assets/cure_tsr_severity_grid.png")
    plot_gate_weights_comparison(results, "assets/cure_tsr_gate_weights.png")
    print("Wrote assets/cure_tsr_severity_grid.png, assets/cure_tsr_gate_weights.png")


if __name__ == "__main__":
    main()
