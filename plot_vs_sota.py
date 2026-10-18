"""
Renders the head-to-head charts from benchmark_vs_sota.py's output.

    python plot_vs_sota.py synthetic   # reads vs_sota_synthetic_results.json
    python plot_vs_sota.py gtsrb
    python plot_vs_sota.py cure

Writes assets/vs_sota_<dataset>_curves.png (accuracy vs. severity, one panel
per sweep, mean with a +/-1 std band across seeds) and
assets/vs_sota_<dataset>_cost.png (accuracy at the hardest level vs. compute).
Also prints a markdown table of the same numbers for the README, since a
chart alone is not an accessible record of the result.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
# Fixed categorical order (never cycled), validated for CVD separation in light mode.
COLORS = {
    "dual_pathway": "#2a78d6",
    "magno_only": "#eb6834",
    "parvo_only": "#1baf7a",
    "resnet18": "#eda100",
    "mobilenet_v3_small": "#e87ba4",
}
MARKERS = {"dual_pathway": "o", "magno_only": "s", "parvo_only": "^", "resnet18": "D", "mobilenet_v3_small": "v"}
LABELS = {
    "dual_pathway": "dual_pathway",
    "magno_only": "magno_only",
    "parvo_only": "parvo_only",
    "resnet18": "ResNet-18",
    "mobilenet_v3_small": "MobileNetV3-S",
}
# Points that cluster at ~200-330 MMACs get labels fanned out to the left.
LABEL_OFFSETS = {
    "resnet18": (-10, 22, "right"),
    "dual_pathway": (-10, 6, "right"),
    "parvo_only": (-10, -14, "right"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    res = json.loads((HERE / f"vs_sota_{ds}_results.json").read_text())
    models = res["models"]
    sweeps = list(next(iter(models.values()))["accuracy"])
    chance = 1.0 / res["num_classes"]

    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.2 * len(sweeps), 3.6), sharey=True)
    axes = axes if len(sweeps) > 1 else [axes]
    for ax, sweep in zip(axes, sweeps):
        _style(ax)
        for name, m in models.items():
            pts = [(i, lv["mean"], lv["std"]) for i, lv in enumerate(m["accuracy"][sweep]) if lv]
            xs, mu, sd = zip(*pts)
            c = COLORS.get(name, MUTED)
            ax.fill_between(xs, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)],
                            color=c, alpha=0.15, linewidth=0)
            ax.plot(xs, mu, color=c, linewidth=2, marker=MARKERS.get(name, "o"), markersize=6,
                    markeredgecolor="white", markeredgewidth=1, label=LABELS.get(name, name))
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(sweep.replace("_", " "), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("severity level (0 = clean)", color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(next(iter(models.values()))["accuracy"][sweep])))
    axes[0].set_ylabel("test accuracy", color=MUTED, fontsize=9)
    axes[0].text(0, chance - 0.05, "chance", color=MUTED, fontsize=8, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    n = len(res["seeds"])
    fig.suptitle(f"{ds}: accuracy vs. capture severity (mean of {n} seeds, band = ±1 std)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = HERE / "assets" / f"vs_sota_{ds}_curves.png"
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out}")

    # Accuracy at the hardest level of the hardest sweep vs. compute. Five
    # entities is past a safe categorical cap for a scatter, so identity is
    # carried by direct labels on one neutral mark color instead of hue.
    hard = "combined" if "combined" in sweeps else sweeps[-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for name, m in models.items():
        lv = [x for x in m["accuracy"][hard] if x][-1]
        mm = max(m["cost"]["mmacs"], 1e-3)
        ax.errorbar(mm, lv["mean"], yerr=lv["std"], fmt="o", color=INK, markersize=7,
                    markeredgecolor="white", ecolor=MUTED, capsize=3)
        dx, dy, ha = LABEL_OFFSETS.get(name, (8, 4, "left"))
        ax.annotate(f"{LABELS.get(name, name)}, {m['cost']['params'] / 1e6:.2f}M params",
                    (mm, lv["mean"]), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("compute per image (MMACs, log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel(f"accuracy, {hard} level {len(m['accuracy'][hard]) - 1}", color=MUTED, fontsize=9)
    ax.set_title(f"{ds}: accuracy at max severity vs. compute", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out = HERE / "assets" / f"vs_sota_{ds}_cost.png"
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out}")

    # Markdown table: per sweep, mean of degraded levels (1..N) and the hardest level.
    print()
    hdr = "| model | params | MMACs | CPU ms | clean | " + " | ".join(
        f"{s} avg lvl 1-{len(models[next(iter(models))]['accuracy'][s]) - 1} | {s} max" for s in sweeps) + " |"
    print(hdr)
    print("|" + "---|" * (5 + 2 * len(sweeps)))
    for name, m in models.items():
        a = m["accuracy"]
        clean = a[sweeps[0]][0]
        row = [LABELS.get(name, name), f"{m['cost']['params'] / 1e6:.2f}M", f"{m['cost']['mmacs']:.1f}",
               f"{m['cost']['cpu_latency_ms']:.1f}", f"{100 * clean['mean']:.1f} ± {100 * clean['std']:.1f}"]
        for s in sweeps:
            lv = [x for x in a[s][1:] if x]
            avg = sum(x["mean"] for x in lv) / len(lv)
            row += [f"{100 * avg:.1f}", f"{100 * lv[-1]['mean']:.1f} ± {100 * lv[-1]['std']:.1f}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
