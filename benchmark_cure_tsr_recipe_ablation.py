"""
Isolates which ingredient of the "GTSRB-style improvement recipe" (more
training data, weight_decay + LR schedule + more epochs, inverse-frequency
class weighting, spatial train-time augmentation) caused CURE-TSR accuracy
to regress when all four were bundled into one change (see README.md /
session history: that attempt dropped dual_pathway, dual_pathway_lateral,
parvo_only, AND magno_only all by ~4-5 points average accuracy, including
the two single-pathway baselines that have nothing to do with fusion or
gating -- so whatever caused it is common to all four, not a fusion-
specific side effect).

Trains ONLY ParvoOnlyClassifier and MagnoOnlyClassifier -- no pathway
dropout, no gate, no lateral connection, nothing stochastic in their own
architecture -- specifically so a regression can't be blamed on gating
dynamics. Five conditions, each changing exactly one thing relative to the
known-good baseline (benchmark_cure_tsr.py's committed recipe: 150/class,
20 epochs, no augmentation, no class weights, no weight_decay/schedule):

  baseline          - the committed recipe, unchanged
  more_data         - only TRAIN_N_PER_CLASS 150 -> 800
  training_regime   - only weight_decay=1e-4 + StepLR(10, 0.5) + 20->30 epochs
  class_weights     - only inverse-frequency weighted CrossEntropyLoss
  augmentation      - only spatial (rotation/scale/translation) train-time jitter

Whichever condition(s) reproduce the regression on their own are the actual
culprit(s); the others should track the baseline.

Run: python benchmark_cure_tsr_recipe_ablation.py (needs cure_tsr_data/,
same as benchmark_cure_tsr.py)
"""

import json
import random
import statistics

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from benchmark_cure_tsr import (
    CHALLENGE_NAMES, CHALLENGES_TO_TEST, CLASS_NAMES, CURE_TSR_ROOT, DEVICE, IMG_SIZE, LEVELS,
    SIGN_IDS, _batch_real_inputs, evaluate, index_dataset, make_batches,
)
from evaluate import MagnoOnlyClassifier, ParvoOnlyClassifier

import numpy as np

BASELINE_N_PER_CLASS = 150
BASELINE_EPOCHS = 20


def _augment_pil(img, rnd):
    """Same spatial-only augmentation as the retracted recipe attempt --
    rotation + scale + translation, no brightness/color jitter (see that
    attempt's reasoning: CURE-TSR's own test challenges probe brightness/
    color axes directly, so jittering those at train time would be
    training partly for the test)."""
    w, h = img.size
    angle = rnd.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(127, 127, 127))
    scale = rnd.uniform(0.85, 1.15)
    img = img.resize((max(4, round(w * scale)), max(4, round(h * scale))), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (127, 127, 127))
    off_x = rnd.randint(-max(1, w // 6), max(1, w // 6))
    off_y = rnd.randint(-max(1, h // 6), max(1, h // 6))
    paste_x = (w - img.size[0]) // 2 + off_x
    paste_y = (h - img.size[1]) // 2 + off_y
    canvas.paste(img, (paste_x, paste_y))
    return canvas


def _load_image(path, augment_rnd=None):
    img = Image.open(path).convert("RGB")
    if augment_rnd is not None:
        img = _augment_pil(img, augment_rnd)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _sample_split(buckets, challenge, level, n_per_class, seed, augment=False):
    rnd = random.Random(seed)
    imgs, labs = [], []
    for label_idx, sign_id in enumerate(SIGN_IDS):
        pool = buckets.get((sign_id, challenge, level), [])
        if not pool:
            continue
        pool = pool[:]
        rnd.shuffle(pool)
        for path in pool[:n_per_class]:
            imgs.append(_load_image(path, augment_rnd=rnd if augment else None))
            labs.append(label_idx)
    if not imgs:
        return None, None
    imgs, labs = torch.stack(imgs), torch.tensor(labs)
    perm = torch.randperm(len(labs))
    return imgs[perm], labs[perm]


def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    for color_imgs, labels in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_real_inputs(color_imgs)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        optimizer.zero_grad()
        logits = model(gray_batch, color_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_and_eval(model_ctor, model_seed, train_imgs, train_labs, class_weights,
                    epochs, weight_decay, use_schedule, buckets, clean_test_loader):
    torch.manual_seed(model_seed)
    model = model_ctor(num_classes=len(SIGN_IDS)).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5) if use_schedule else None
    train_loader = make_batches(train_imgs, train_labs)

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, DEVICE, class_weights=class_weights)
        if scheduler is not None:
            scheduler.step()

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
    avg_acc = statistics.mean(v for curve in severity_curve.values() for v in curve.values())
    return avg_acc, severity_curve


def main():
    if not CURE_TSR_ROOT.exists():
        raise SystemExit(f"{CURE_TSR_ROOT} not found — see benchmark_cure_tsr.py's docstring.")

    num_classes = len(SIGN_IDS)
    print(f"Indexing CURE-TSR under {CURE_TSR_ROOT} ({num_classes} sign classes)...")
    buckets = index_dataset(CURE_TSR_ROOT)
    if not buckets:
        raise SystemExit("No matching *.bmp files found under cure_tsr_data/.")

    clean_test_imgs, clean_test_labs = _sample_split(buckets, challenge=0, level=0, n_per_class=40, seed=1)
    clean_test_loader = make_batches(clean_test_imgs, clean_test_labs)

    conditions = {
        "baseline":        dict(n_per_class=BASELINE_N_PER_CLASS, epochs=BASELINE_EPOCHS,
                                 weight_decay=0.0, schedule=False, class_weight=False, augment=False),
        "more_data":       dict(n_per_class=800, epochs=BASELINE_EPOCHS,
                                 weight_decay=0.0, schedule=False, class_weight=False, augment=False),
        "training_regime": dict(n_per_class=BASELINE_N_PER_CLASS, epochs=30,
                                 weight_decay=1e-4, schedule=True, class_weight=False, augment=False),
        "class_weights":   dict(n_per_class=BASELINE_N_PER_CLASS, epochs=BASELINE_EPOCHS,
                                 weight_decay=0.0, schedule=False, class_weight=True, augment=False),
        "augmentation":    dict(n_per_class=BASELINE_N_PER_CLASS, epochs=BASELINE_EPOCHS,
                                 weight_decay=0.0, schedule=False, class_weight=False, augment=True),
    }

    results = {}
    for cond_name, cfg in conditions.items():
        print(f"\n########## Condition: {cond_name} ({cfg}) ##########")
        train_imgs, train_labs = _sample_split(
            buckets, challenge=0, level=0, n_per_class=cfg["n_per_class"], seed=0, augment=cfg["augment"])
        print(f"  train images: {len(train_labs)}")

        class_weights = None
        if cfg["class_weight"]:
            counts = torch.bincount(train_labs, minlength=num_classes).float()
            class_weights = (counts.sum() / (num_classes * counts.clamp(min=1))).to(DEVICE)

        results[cond_name] = {}
        for model_name, ctor, seed in (("magno_only", MagnoOnlyClassifier, 9001),
                                        ("parvo_only", ParvoOnlyClassifier, 9002)):
            avg_acc, curve = train_and_eval(
                ctor, seed, train_imgs, train_labs, class_weights,
                cfg["epochs"], cfg["weight_decay"], cfg["schedule"], buckets, clean_test_loader)
            results[cond_name][model_name] = {"avg_all_levels": avg_acc, "severity_curve": curve}
            print(f"  {model_name:12s} avg_all_levels_accuracy={avg_acc:.3f}")

    print("\n\n========== SUMMARY (delta vs. baseline) ==========")
    base = results["baseline"]
    print(f"  {'condition':18s} {'magno_only':>12s} {'parvo_only':>12s}")
    for cond_name in conditions:
        m = results[cond_name]["magno_only"]["avg_all_levels"]
        p = results[cond_name]["parvo_only"]["avg_all_levels"]
        dm = m - base["magno_only"]["avg_all_levels"]
        dp = p - base["parvo_only"]["avg_all_levels"]
        print(f"  {cond_name:18s} {m:8.3f} ({dm:+.3f}) {p:8.3f} ({dp:+.3f})")

    with open("cure_tsr_recipe_ablation_results.json", "w") as f:
        json.dump({"classes": CLASS_NAMES, "conditions": conditions, "results": results}, f, indent=2)
    print("\nWrote cure_tsr_recipe_ablation_results.json")


if __name__ == "__main__":
    main()
