"""
Renders assets/gtsrb_task_groups.png: accuracy split by whether the task
group structurally needs parvo (digit-only speed-limit signs) or is
solvable from coarse cues alone (shape-distinct signs). Run
benchmark_gtsrb_full.py first (it writes the grouped numbers into
gtsrb_full_results.json).
"""

import json

import matplotlib.pyplot as plt

from plot_results import AQUA, BLUE, INK_PRIMARY, INK_SECONDARY, ORANGE, _clean_axes


def main():
    with open("gtsrb_full_results.json") as f:
        results = json.load(f)

    models = [("dual_pathway", BLUE), ("parvo_only", ORANGE), ("magno_only", AQUA)]
    groups = ["digit_only", "shape_distinct"]
    group_labels = ["Digit-only signs\n(shape/color identical — parvo required)",
                     "Shape-distinct signs\n(coarse cues suffice)"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    _clean_axes(ax)

    n_models = len(models)
    width = 0.8 / n_models
    x = range(len(groups))
    for i, (name, color) in enumerate(models):
        vals = [results[f"clean_accuracy_{g}"][name] for g in groups]
        xs = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        ax.bar(xs, vals, width=width, color=color, label=name, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy (official GTSRB test set)")
    ax.set_title("Accuracy by task group: does this task actually need parvo?",
                 color=INK_PRIMARY, fontsize=12.5, loc="left", pad=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    fig.text(0.02, -0.03,
              "magno_only scores nearly the same on both groups (90.8% vs 90.8%) — at its trained\n"
              "16x16 resolution it isn't actually blind to the digit-only group as hypothesized.",
              fontsize=8.5, color=INK_SECONDARY, ha="left")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("assets/gtsrb_task_groups.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/gtsrb_task_groups.png")


if __name__ == "__main__":
    main()
