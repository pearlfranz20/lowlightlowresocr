"""
Controlled, multi-seed A/B for the lateral gate-collapse question (see
README.md, "Learned gate weighting, clean vs. heavily degraded" under the
CURE-TSR results section, and dual_pathway_classifier.py's LateralParvoStream
docstring).

benchmark_cure_tsr.py's own before/after (two separate full runs of all four
models) wasn't a clean isolated test: pathway-dropout and lateral-dropout
coin flips all drew from the same shared global `random` module, and since
dual_pathway_lateral trains before parvo_only/magno_only each epoch, giving
it lateral_dropout_p > 0 shifted the *other* models' dropout draws too
(visible as parvo_only's own accuracy moving between those two runs, despite
its code being untouched).

This script isolates it properly, per seed pair: trains
DualPathwayLateralClassifier twice, back to back --

  arm A: lateral_dropout_p=0.0  (unconditional lateral broadcast, i.e. the
         behavior before this flag existed at all)
  arm B: lateral_dropout_p=0.3  (the proposed fix)

-- with the SAME torch weight-init seed and the SAME private pathway-dropout
RNG sequence for both arms in a given seed pair (see train_epoch's
dropout_rng and LateralParvoStream's rng_seed params). Neither arm touches
any other model, and both arms see the exact same train/test data splits.
Within one seed pair, the only thing that can differ between arms is
whether the lateral broadcast got randomly skipped during training.

A single seed pair isn't enough to trust, though: the first controlled run
(seed pair 0) showed arm A (no dropout) NOT collapsing on its own -- a
healthy 81/19 clean gate split -- and beating arm B on every single curve,
including clean accuracy. That directly contradicts the original
(confounded) run that motivated this fix in the first place, where arm-A
equivalent behavior fully collapsed to 100% parvo. Either the collapse is a
real but seed-sensitive failure mode (common enough to be worth guarding
against on average, rare enough that any one run can miss it), or it was a
one-off fluke and the "fix" is actually a net-negative regularization tax.
Only repeating across several independent seed pairs can tell those apart,
which is what this script now does -- reports per-seed results AND
aggregate stats (mean/std of accuracy, and how often each arm's gate
lands in a near-collapsed state) across all of them.

Run: python benchmark_cure_tsr_ablation.py  (after benchmark_cure_tsr.py's
usual CURE-TSR download/extraction under cure_tsr_data/)
"""

import json
import random
import statistics

import torch
from torch.optim import Adam

from benchmark_cure_tsr import (
    CHALLENGE_NAMES, CHALLENGES_TO_TEST, CLASS_NAMES, CURE_TSR_ROOT, DEVICE, LEVELS,
    SIGN_IDS, _sample_split, avg_gate_weights, evaluate, index_dataset, make_batches,
    train_epoch,
)
from dual_pathway_classifier import DualPathwayLateralClassifier

# (model_init_seed, dropout_rng_seed, lateral_rng_seed) per repeat. Three
# independent draws -- enough to tell "always collapses" / "never collapses"
# / "sometimes collapses" apart without an unreasonable CPU training budget
# (6 model trainings total: 3 seed pairs x 2 arms).
SEED_PAIRS = [(5000, 6000, 7000), (5100, 6100, 7100), (5200, 6200, 7200)]
EPOCHS = 20
PATHWAY_DROPOUT_P = 0.15
# a gate that never moves more than this far from 0 or 1 (on either the
# clean or degraded average) is considered "collapsed" for the summary --
# not a hard architectural threshold, just a legible reporting cutoff.
COLLAPSE_THRESHOLD = 0.05


def train_arm(lateral_dropout_p, model_seed, dropout_seed, lateral_seed, train_loader, num_classes):
    torch.manual_seed(model_seed)
    model = DualPathwayLateralClassifier(
        num_classes=num_classes, with_text_head=False,
        lateral_dropout_p=lateral_dropout_p, lateral_rng_seed=lateral_seed,
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    dropout_rng = random.Random(dropout_seed)

    losses = []
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, DEVICE,
                            pathway_dropout_p=PATHWAY_DROPOUT_P, dropout_rng=dropout_rng)
        losses.append(loss)
    return model, losses


def evaluate_arm(model, buckets, clean_test_loader):
    severity_curve = {CHALLENGE_NAMES[c]: {} for c in CHALLENGES_TO_TEST}
    for challenge in CHALLENGES_TO_TEST:
        cname = CHALLENGE_NAMES[challenge]
        for level in LEVELS:
            if level == 0:
                loader = clean_test_loader
            else:
                imgs, labs = _sample_split(buckets, challenge=challenge, level=level, n_per_class=40, seed=1)
                loader = make_batches(imgs, labs) if imgs is not None else None
            if loader is None:
                continue
            severity_curve[cname][level] = evaluate(model, loader, DEVICE)

    clean_gate = avg_gate_weights(model, clean_test_loader, DEVICE)
    degraded_samples = []
    for challenge in CHALLENGES_TO_TEST:
        imgs, labs = _sample_split(buckets, challenge=challenge, level=5, n_per_class=40, seed=1)
        if imgs is None:
            continue
        degraded_samples.append(avg_gate_weights(model, make_batches(imgs, labs), DEVICE))
    degraded_gate = torch.stack(degraded_samples).mean(dim=0) if degraded_samples else clean_gate

    return {
        "severity_curve": severity_curve,
        "gate_weights": {
            "clean": [clean_gate[0].item(), clean_gate[1].item()],
            "degraded": [degraded_gate[0].item(), degraded_gate[1].item()],
        },
    }


def _avg_all_levels(severity_curve):
    return statistics.mean(v for curve in severity_curve.values() for v in curve.values())


def _is_collapsed(gate_weights):
    vals = gate_weights["clean"] + gate_weights["degraded"]
    return any(v < COLLAPSE_THRESHOLD or v > 1 - COLLAPSE_THRESHOLD for v in vals)


def main():
    if not CURE_TSR_ROOT.exists():
        raise SystemExit(f"{CURE_TSR_ROOT} not found — see benchmark_cure_tsr.py's docstring.")

    num_classes = len(SIGN_IDS)
    print(f"Indexing CURE-TSR under {CURE_TSR_ROOT} ({num_classes} sign classes)...")
    buckets = index_dataset(CURE_TSR_ROOT)
    if not buckets:
        raise SystemExit("No matching *.bmp files found under cure_tsr_data/.")

    train_imgs, train_labs = _sample_split(buckets, challenge=0, level=0, n_per_class=150, seed=0)
    train_loader = make_batches(train_imgs, train_labs)
    clean_test_imgs, clean_test_labs = _sample_split(buckets, challenge=0, level=0, n_per_class=40, seed=1)
    clean_test_loader = make_batches(clean_test_imgs, clean_test_labs)
    print(f"Train: {len(train_labs)} images  |  {len(SEED_PAIRS)} independent seed pairs x 2 arms")

    all_results = {"no_lateral_dropout": [], "with_lateral_dropout_0.3": []}

    for pair_idx, (model_seed, dropout_seed, lateral_seed) in enumerate(SEED_PAIRS):
        print(f"\n########## Seed pair {pair_idx} (model={model_seed}, dropout={dropout_seed}) ##########")
        for arm_name, p in (("no_lateral_dropout", 0.0), ("with_lateral_dropout_0.3", 0.3)):
            print(f"\n=== Arm: {arm_name} (lateral_dropout_p={p}) ===")
            model, losses = train_arm(p, model_seed, dropout_seed, lateral_seed, train_loader, num_classes)
            print("  final train loss:", f"{losses[-1]:.3f}")
            result = evaluate_arm(model, buckets, clean_test_loader)
            gw = result["gate_weights"]
            avg_acc = _avg_all_levels(result["severity_curve"])
            collapsed = _is_collapsed(gw)
            print(f"  gate: clean magno={gw['clean'][0]:.3f} parvo={gw['clean'][1]:.3f}  |  "
                  f"degraded magno={gw['degraded'][0]:.3f} parvo={gw['degraded'][1]:.3f}"
                  f"{'  [COLLAPSED]' if collapsed else ''}")
            print(f"  avg_all_levels_accuracy={avg_acc:.3f}")
            all_results[arm_name].append({
                "seed_pair": pair_idx, "severity_curve": result["severity_curve"],
                "gate_weights": gw, "avg_all_levels_accuracy": avg_acc, "collapsed": collapsed,
            })

    print("\n\n========== AGGREGATE (across all seed pairs) ==========")
    summary = {}
    for arm_name, runs in all_results.items():
        accs = [r["avg_all_levels_accuracy"] for r in runs]
        n_collapsed = sum(r["collapsed"] for r in runs)
        clean_magno = [r["gate_weights"]["clean"][0] for r in runs]
        degraded_magno = [r["gate_weights"]["degraded"][0] for r in runs]
        summary[arm_name] = {
            "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
            "acc_per_seed": accs,
            "n_collapsed": n_collapsed, "n_total": len(runs),
            "clean_magno_mean": statistics.mean(clean_magno),
            "degraded_magno_mean": statistics.mean(degraded_magno),
        }
        print(f"  {arm_name:24s} acc={summary[arm_name]['acc_mean']:.3f} "
              f"(+/-{summary[arm_name]['acc_std']:.3f}, per-seed={[round(a,3) for a in accs]})  "
              f"collapsed={n_collapsed}/{len(runs)}  "
              f"gate magno clean/degraded={summary[arm_name]['clean_magno_mean']:.3f}/"
              f"{summary[arm_name]['degraded_magno_mean']:.3f}")

    with open("cure_tsr_ablation_results.json", "w") as f:
        json.dump({"classes": CLASS_NAMES, "seed_pairs": SEED_PAIRS,
                    "per_seed_results": all_results, "summary": summary}, f, indent=2)
    print("\nWrote cure_tsr_ablation_results.json")


if __name__ == "__main__":
    main()
