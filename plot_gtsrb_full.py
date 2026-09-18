"""
Renders assets/gtsrb_full_comparison.png: this repo's full-43-class GTSRB
run against published literature numbers. Run benchmark_gtsrb_full.py first.

Literature figures (see README citation): IJCNN 2011 competition winner
99.46%, human performance 98.84%, current SOTA (ConvNeSe) 99.85% — sourced
from https://paperswithcode.com/sota/traffic-sign-recognition-on-gtsrb and
the original competition paper, not derived from this repo's own runs.
"""

import json

import matplotlib.pyplot as plt

from plot_results import (
    AQUA,
    BASELINE,
    BLUE,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORANGE,
    SURFACE,
    _clean_axes,
)

VIOLET = "#4a3aa7"  # literature reference bars — distinct from our pathway colors


def main():
    with open("gtsrb_full_results.json") as f:
        results = json.load(f)
    acc = results["clean_accuracy"]
    n_train, epochs = results["train_images"], results["epochs"]

    bars = [
        ("dual_pathway\n(this repo)", acc["dual_pathway"], BLUE),
        ("parvo_only\n(this repo)", acc["parvo_only"], ORANGE),
        ("magno_only\n(this repo)", acc["magno_only"], AQUA),
        ("GTSRB\nhuman perf.", 0.9884, VIOLET),
        ("GTSRB 2011\ncomp. winner", 0.9946, VIOLET),
        ("GTSRB\ncurrent SOTA", 0.9985, VIOLET),
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    _clean_axes(ax)

    xs = range(len(bars))
    for x, (label, val, color) in zip(xs, bars):
        ax.bar(x, val, color=color, width=0.6, zorder=3)
        ax.text(x, val + 0.015, f"{val:.1%}", ha="center", fontsize=10, color=INK_SECONDARY)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy (official GTSRB test set, all 43 classes)")
    ax.set_title("This repo vs. published GTSRB results (clean accuracy)",
                 color=INK_PRIMARY, fontsize=12.5, loc="left", pad=14)
    ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)

    fig.text(0.02, -0.02,
              f"This repo: {n_train:,} augmented train images, {epochs} epochs, untuned toy architecture.\n"
              "Literature: full ~35k train images, heavy augmentation, tuned/ensembled architectures.",
              fontsize=8.5, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("assets/gtsrb_full_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/gtsrb_full_comparison.png")


if __name__ == "__main__":
    main()
