"""
Head-to-head: the dual-pathway model vs. standard modern CNN backbones
(ResNet-18, MobileNetV3-Small) under realistic low-light / low-resolution
capture, with a protocol designed to be fair to every model.

Why this script exists
----------------------
The earlier severity sweeps (benchmark.py, benchmark_gtsrb.py) corrupted
only the parvo input and always derived magno from the *clean* image, so
any model with a magno pathway had access to uncorrupted information the
parvo-only baseline never saw. A real camera produces one degraded frame,
and both pathways have to be derived from it. This script fixes that:

  1. One degraded "capture" per sample (simulate_capture), THEN both the
     gray low-res magno input and the color parvo input are derived from
     it. No pathway ever sees the clean image at test time.
  2. Every model gets the same data, the same degradation-aware training
     (random capture conditions per sample), the same optimizer, schedule,
     epochs, and weight decay. Pathway dropout is the dual model's own
     architecture-specific regularizer and is the only difference.
  3. Multiple seeds, reported as mean and std, plus model cost (params,
     MACs at the benchmark's input size, single-image CPU latency), because
     "on par" with a backbone 50x larger is a different claim than "on par"
     with one the same size.

Capture model (simulate_capture)
--------------------------------
  * low resolution: area-downsample by `res` then bilinear-upsample back to
    the working size (a sign seen from further away covers fewer pixels).
  * low light: photon shot noise (Poisson, FULL_WELL photons at full white
    and light=1), Gaussian read noise, 8-bit quantization at the sensor's
    exposure level, then digital gain back up by 1/light (what auto-
    exposure on a phone or dashcam does). Dark frames therefore have both
    amplified noise and very few distinct code values, as real ones do.

Three sweeps are evaluated: light only, resolution only, and both combined
(the "night, far away" case this repo is named for).

Datasets
--------
  --dataset synthetic   PIL-rendered signs from benchmark.py; runs offline.
  --dataset gtsrb       GTSRB training set split by track (no near-duplicate
                        leakage) + the official test set. Same files as
                        benchmark_gtsrb_full.py expects.
  --dataset cure        CURE-TSR real sequences, using the release's own
                        train/test folders (see load_cure_tsr). Train on
                        challenge-free only (the paper's Experiment 1
                        protocol) and evaluate on CURE-TSR's own rendered
                        challenges instead of simulate_capture.

Usage
-----
    python benchmark_vs_sota.py --dataset synthetic --seeds 0 1 2
    python benchmark_vs_sota.py --dataset gtsrb --seeds 0 1 2 --epochs 20
    python benchmark_vs_sota.py --dataset cure --seeds 0 1 2
Writes vs_sota_<dataset>_results.json; plot with plot_vs_sota.py.
"""

import argparse
import csv
import json
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from dual_pathway_classifier import DualPathwayClassifier
from evaluate import MagnoOnlyClassifier, ParvoOnlyClassifier

HERE = Path(__file__).parent
DEVICE = "cpu"
MAGNO_SIZE = 16
FULL_WELL = 1000.0   # photons at full white, light=1.0
READ_NOISE = 3.0     # electrons, Gaussian

# (light fraction, resolution fraction) per level, level 0 is clean.
SWEEPS = {
    "low_light": [(1.0, 1.0), (0.1, 1.0), (0.03, 1.0), (0.01, 1.0), (0.003, 1.0)],
    "low_res":   [(1.0, 1.0), (1.0, 0.5), (1.0, 0.33), (1.0, 0.25), (1.0, 0.18)],
    "combined":  [(1.0, 1.0), (0.1, 0.5), (0.03, 0.33), (0.01, 0.25), (0.003, 0.18)],
}


# --------------------------------------------------------------------------
# Capture simulation
# --------------------------------------------------------------------------
def simulate_capture(img, light=1.0, res=1.0, gen=None):
    """img: (3, H, W) in [0, 1]. Returns one degraded frame, same shape."""
    out = img
    if res < 1.0:
        h, w = img.shape[-2:]
        small = (max(2, round(h * res)), max(2, round(w * res)))
        out = F.interpolate(out.unsqueeze(0), size=small, mode="area")
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    if light < 1.0:
        n = light * FULL_WELL
        electrons = torch.poisson(out.clamp(0, 1) * n, generator=gen)
        electrons = electrons + torch.randn(out.shape, generator=gen) * READ_NOISE
        raw = (electrons / FULL_WELL).clamp(0, 1)          # what the sensor records
        raw = torch.round(raw * 255.0) / 255.0             # 8-bit ADC at that exposure
        out = (raw / light).clamp(0, 1)                    # digital gain (auto-exposure)
    return out


def random_capture(img, gen):
    """Training-time condition sampler: log-uniform light in [0.002, 1],
    uniform res in [0.15, 1]; one third of samples kept clean."""
    u = torch.rand(3, generator=gen)
    if u[0] < 1 / 3:
        return img
    light = float(10 ** (-2.7 * u[1]))
    res = float(0.15 + 0.85 * u[2])
    return simulate_capture(img, light, res, gen)


def split_pathways(frames):
    """Both pathways derived from the same (possibly degraded) frames."""
    gray = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]).unsqueeze(1)
    gray = F.interpolate(gray, size=(MAGNO_SIZE, MAGNO_SIZE), mode="area")
    return gray, frames


# --------------------------------------------------------------------------
# Models: every model takes (gray, color) so one loop drives all of them
# --------------------------------------------------------------------------
class TorchvisionWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, gray=None, color=None):
        return self.net(color)


def build_model(name, num_classes):
    import torchvision.models as tvm
    if name == "dual_pathway":
        return DualPathwayClassifier(num_classes, with_text_head=False)
    if name == "magno_only":
        return MagnoOnlyClassifier(num_classes)
    if name == "parvo_only":
        return ParvoOnlyClassifier(num_classes)
    if name == "resnet18":
        return TorchvisionWrapper(tvm.resnet18(num_classes=num_classes))
    if name == "mobilenet_v3_small":
        return TorchvisionWrapper(tvm.mobilenet_v3_small(num_classes=num_classes))
    raise ValueError(name)


MODEL_NAMES = ["dual_pathway", "magno_only", "parvo_only", "resnet18", "mobilenet_v3_small"]


@torch.no_grad()
def model_cost(model, img_size):
    model.eval()
    gray = torch.zeros(1, 1, MAGNO_SIZE, MAGNO_SIZE)
    color = torch.zeros(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as fc:
        model(gray, color)
    macs = fc.get_total_flops() / 2
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for _ in range(5):
        model(gray, color)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        model(gray, color)
        times.append(time.perf_counter() - t0)
    torch.set_num_threads(threads)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "mmacs": macs / 1e6,
        "cpu_latency_ms": 1000 * statistics.median(times),
    }


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def train(model, name, imgs, labs, epochs, seed, degrade, batch_size=32, pathway_dropout_p=0.15):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    rng = random.Random(seed + 29)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * ((len(labs) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(labs), generator=gen)
        total = 0.0
        for i in range(0, len(labs), batch_size):
            idx = perm[i:i + batch_size]
            frames = imgs[idx]
            if degrade:
                frames = torch.stack([random_capture(f, gen) for f in frames])
            gray, color = split_pathways(frames)
            if name == "dual_pathway":
                r = rng.random()
                if r < pathway_dropout_p / 2:
                    color = torch.zeros_like(color)
                elif r < pathway_dropout_p:
                    gray = torch.zeros_like(gray)
            opt.zero_grad()
            loss = crit(model(gray, color), labs[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"    {name:20s} epoch {epoch + 1:3d}  loss={total / len(labs):.3f}", flush=True)
    return model


@torch.no_grad()
def accuracy(model, frames, labs, batch_size=128):
    model.eval()
    correct = 0
    for i in range(0, len(labs), batch_size):
        gray, color = split_pathways(frames[i:i + batch_size])
        correct += (model(gray, color).argmax(1) == labs[i:i + batch_size]).sum().item()
    return correct / len(labs)


@torch.no_grad()
def degraded_test_sets(test_imgs, eval_seed=12345):
    """Pre-render every (sweep, level) test set once with a fixed generator,
    so every model and every seed is scored on the exact same frames."""
    sets = {}
    for sweep, levels in SWEEPS.items():
        sets[sweep] = []
        for light, res in levels:
            gen = torch.Generator().manual_seed(eval_seed)
            sets[sweep].append(torch.stack([simulate_capture(im, light, res, gen) for im in test_imgs]))
    return sets


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def load_synthetic():
    from benchmark import build_dataset, CLASSES, IMG_SIZE
    tr_x, tr_y, _ = build_dataset(n_per_class=150, seed=0)
    te_x, te_y, _ = build_dataset(n_per_class=100, seed=1)
    return tr_x, tr_y, te_x, te_y, CLASSES, IMG_SIZE


GTSRB_SIZE = 48


def _gtsrb_crop(path, r):
    box = tuple(int(r[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))
    img = Image.open(path).convert("RGB").crop(box).resize((GTSRB_SIZE, GTSRB_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_gtsrb(frames_per_track=10, test_cap_per_class=None, seed=0):
    """No autocontrast here on purpose: contrast stretching is part of the
    camera pipeline after capture, and simulate_capture already applies the
    digital gain a real auto-exposure would."""
    from benchmark_gtsrb import GTSRB_ROOT
    test_dir = HERE / "gtsrb_data" / "GTSRB" / "Final_Test" / "Images"
    test_csv = HERE / "gtsrb_data" / "GTSRB_Test_GT" / "GT-final_test.csv"
    rnd = random.Random(seed)
    tr_x, tr_y = [], []
    for c in range(43):
        folder = GTSRB_ROOT / f"{c:05d}"
        with open(folder / f"GT-{c:05d}.csv", newline="") as f:
            recs = list(csv.DictReader(f, delimiter=";"))
        tracks = defaultdict(list)
        for r in recs:
            tracks[r["Filename"].split("_")[0]].append(r)
        for frames in tracks.values():
            rnd.shuffle(frames)
            for r in frames[:frames_per_track]:
                tr_x.append(_gtsrb_crop(folder / r["Filename"], r))
                tr_y.append(c)
    with open(test_csv, newline="") as f:
        recs = list(csv.DictReader(f, delimiter=";"))
    if test_cap_per_class:
        by_c = defaultdict(list)
        for r in recs:
            by_c[int(r["ClassId"])].append(r)
        recs = [r for rs in by_c.values() for r in rnd.sample(rs, min(len(rs), test_cap_per_class))]
    te_x = [_gtsrb_crop(test_dir / r["Filename"], r) for r in recs]
    te_y = [int(r["ClassId"]) for r in recs]
    return (torch.stack(tr_x), torch.tensor(tr_y), torch.stack(te_x), torch.tensor(te_y),
            [str(i) for i in range(43)], GTSRB_SIZE)


CURE_SIZE = 64
CURE_CHALLENGES = {1: "decolorization", 4: "darkening", 6: "exposure", 7: "gaussian_blur",
                   8: "noise", 12: "haze"}
_CURE_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.bmp$", re.IGNORECASE)


def _cure_split_of(path):
    parts = [p.lower() for p in path.parts]
    if any("test" in p for p in parts):
        return "test"
    if any("train" in p for p in parts):
        return "train"
    return None


def load_cure_tsr(root=HERE / "cure_tsr_data", max_train_per_class=None, max_test_per_class=200, seed=0):
    """Uses the release's own train/test folders (any directory whose name
    contains 'train' or 'test'). The original benchmark_cure_tsr.py ignored
    directories and drew train and test from one shuffled pool, so test
    images could be the same photos (or adjacent video frames) as training
    images. That split is not reproduced here: if no train/test folders
    are found, this raises instead of guessing."""
    buckets = defaultdict(list)  # (split, sign, challenge, level) -> paths
    for p in root.rglob("*.bmp"):
        m = _CURE_RE.match(p.name)
        if not m:
            continue
        seq, sign, ch, lvl, _ = (int(g) for g in m.groups())
        split = _cure_split_of(p.relative_to(root))
        if seq != 1 or split is None:
            continue
        buckets[(split, sign, ch, lvl)].append(p)
    if not any(k[0] == "test" for k in buckets):
        raise SystemExit("No CURE-TSR train/test folders found under cure_tsr_data/. Extract the "
                         "Real_Train and Real_Test archives into separate folders.")
    signs = sorted({k[1] for k in buckets})
    rnd = random.Random(seed)

    def load(paths, cap):
        paths = sorted(paths)
        if cap and len(paths) > cap:
            paths = rnd.sample(paths, cap)
        out = []
        for p in paths:
            img = Image.open(p).convert("RGB").resize((CURE_SIZE, CURE_SIZE), Image.BILINEAR)
            out.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1))
        return out

    tr_x, tr_y = [], []
    for li, s in enumerate(signs):
        xs = load(buckets[("train", s, 0, 0)], max_train_per_class)
        tr_x += xs
        tr_y += [li] * len(xs)
    tests = {}
    for ch, cname in CURE_CHALLENGES.items():
        tests[cname] = []
        for lvl in range(6):
            xs, ys = [], []
            for li, s in enumerate(signs):
                key = ("test", s, 0, 0) if lvl == 0 else ("test", s, ch, lvl)
                got = load(buckets[key], max_test_per_class)
                xs += got
                ys += [li] * len(got)
            tests[cname].append((torch.stack(xs), torch.tensor(ys)) if xs else (None, None))
    return torch.stack(tr_x), torch.tensor(tr_y), tests, [str(s) for s in signs], CURE_SIZE


# --------------------------------------------------------------------------
def summarize(per_seed):
    vals = [v for v in per_seed if v is not None]
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "per_seed": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "gtsrb", "cure"], default="synthetic")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--train-regime", choices=["aware", "clean"], default=None,
                    help="aware: random capture degradation in training (default for synthetic/"
                         "gtsrb). clean: challenge-free training only (default for cure, matching "
                         "the CURE-TSR paper's Experiment 1).")
    ap.add_argument("--gtsrb-test-cap", type=int, default=None,
                    help="cap official-test images per class (speed knob for CPU runs)")
    args = ap.parse_args()
    regime = args.train_regime or ("clean" if args.dataset == "cure" else "aware")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print(f"Loading {args.dataset}...", flush=True)
    if args.dataset == "cure":
        tr_x, tr_y, tests, classes, size = load_cure_tsr()
        test_sets = {c: [x for x, _ in lv] for c, lv in tests.items()}
        test_labels = {c: [y for _, y in lv] for c, lv in tests.items()}
    else:
        if args.dataset == "synthetic":
            tr_x, tr_y, te_x, te_y, classes, size = load_synthetic()
        else:
            tr_x, tr_y, te_x, te_y, classes, size = load_gtsrb(test_cap_per_class=args.gtsrb_test_cap)
        print("Rendering degraded test sets...", flush=True)
        test_sets = degraded_test_sets(te_x)
        test_labels = {s: [te_y] * len(v) for s, v in test_sets.items()}
    print(f"  train={len(tr_y)}  classes={len(classes)}  input={size}px  regime={regime}", flush=True)

    results = {"dataset": args.dataset, "train_regime": regime, "epochs": args.epochs,
               "seeds": args.seeds, "num_classes": len(classes), "img_size": size,
               "num_train": len(tr_y), "sweeps": {k: v for k, v in SWEEPS.items()}
               if args.dataset != "cure" else list(test_sets), "models": {}}

    for name in args.models:
        print(f"\n== {name}", flush=True)
        cost = model_cost(build_model(name, len(classes)), size)
        print(f"   params={cost['params']:,}  MMACs={cost['mmacs']:.1f}  "
              f"latency={cost['cpu_latency_ms']:.2f}ms", flush=True)
        acc = {s: [[] for _ in lv] for s, lv in test_sets.items()}
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = build_model(name, len(classes))
            t0 = time.time()
            train(model, name, tr_x, tr_y, args.epochs, seed, degrade=(regime == "aware"))
            for s, levels in test_sets.items():
                for li, frames in enumerate(levels):
                    acc[s][li].append(None if frames is None else accuracy(model, frames, test_labels[s][li]))
            line = "  ".join(f"{s}:" + "/".join("--" if a[-1] is None else f"{a[-1]:.2f}" for a in acc[s])
                             for s in acc)
            print(f"   seed {seed} ({time.time() - t0:.0f}s)  {line}", flush=True)
        results["models"][name] = {
            "cost": cost,
            "accuracy": {s: [summarize(lv) for lv in levels] for s, levels in acc.items()},
        }
        out = HERE / f"vs_sota_{args.dataset}_results.json"
        out.write_text(json.dumps(results, indent=2))  # checkpoint after every model
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
