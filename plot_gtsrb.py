"""Renders assets/gtsrb_*.png from gtsrb_results.json. Run benchmark_gtsrb.py first."""

import json

from plot_results import plot_gate_weights, plot_severity_curve


def main():
    with open("gtsrb_results.json") as f:
        results = json.load(f)

    plot_severity_curve(
        results, "assets/gtsrb_severity_curve.png",
        title="Accuracy vs. corruption severity (held-out GTSRB subset)",
    )
    plot_gate_weights(results, "assets/gtsrb_gate_weights.png")
    print("Wrote assets/gtsrb_severity_curve.png, assets/gtsrb_gate_weights.png")


if __name__ == "__main__":
    main()
