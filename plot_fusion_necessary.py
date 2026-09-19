"""
Renders assets/fusion_necessary.png from fusion_necessary_results.json.
Run benchmark.py first (it runs this experiment as its second phase).
"""

import json

import matplotlib.pyplot as plt

from plot_results import AQUA, BASELINE, BLUE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ORANGE, _clean_axes


def main():
    with open("fusion_necessary_results.json") as f:
        results = json.load(f)
    acc = results["accuracy"]

    bars = [("dual_pathway", acc["dual_pathway"], BLUE),
            ("parvo_only", acc["parvo_only"], ORANGE),
            ("magno_only", acc["magno_only"], AQUA)]

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=200)
    _clean_axes(ax)

    xs = range(len(bars))
    for x, (label, val, color) in zip(xs, bars):
        ax.bar(x, val, color=color, width=0.55, zorder=3)
        ax.text(x, val + 0.02, f"{val:.1%}", ha="center", fontsize=11, color=INK_SECONDARY)

    ax.axhline(0.5, color=BASELINE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax.text(2.35, 0.505, "~chance ceiling for a\nsingle pathway here", fontsize=8, color=INK_MUTED, va="bottom")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (5-draw avg.)")
    ax.set_title("A task neither pathway can solve alone",
                 color=INK_PRIMARY, fontsize=12.5, loc="left", pad=14)

    fig.text(0.02, 0.01,
              "Every test sample has exactly one whole pathway zeroed (chosen per-sample, ~50/50) --\n"
              "not blur, a hard either/or. Only a model that detects which pathway is live and routes\n"
              "to it can beat ~50%; GatedFusion's per-sample weighting is built for exactly this.",
              fontsize=8.5, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0.02, 0.14, 1, 1))
    fig.savefig("assets/fusion_necessary.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/fusion_necessary.png")


if __name__ == "__main__":
    main()
