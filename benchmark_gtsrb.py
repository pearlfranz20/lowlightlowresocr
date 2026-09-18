"""
Real-data version of benchmark.py, using a subset of GTSRB (German Traffic
Sign Recognition Benchmark) instead of the synthetic PIL dataset.

Not run automatically / not part of the package: GTSRB isn't shipped with
this repo (it's ~270MB). Download GTSRB_Final_Training_Images.zip from
https://benchmark.ini.rub.de/gtsrb_news.html (or the archive mirror at
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)
and extract GTSRB/Final_Training/Images/<class>/ into gtsrb_data/ next to
this file before running.

Uses a 10-class subset (4 speed-limit signs that differ only by digits —
the real-world version of this repo's synthetic speed_25/speed_45 pair —
plus 6 shape-distinct classes) rather than the full 43, and splits by
GTSRB "track" (each physical sign was photographed as a ~30-frame burst;
frames within a track are near-duplicates, so a random per-image split
would leak) into train/test itself, since this repo doesn't use the
official separately-released test-label CSV.

No text-region masks exist in GTSRB, so this trains DualPathwayClassifier
with with_text_head=False — classification + gate-weight behavior only,
no text-detection chart for this one.

Run: python benchmark_gtsrb.py
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam

from benchmark import _batch_inputs, avg_gate_weights, severity_curve
from dual_pathway_classifier import DualPathwayClassifier
from evaluate import MagnoOnlyClassifier, ParvoOnlyClassifier, print_comparison_table

GTSRB_ROOT = Path(__file__).parent / "gtsrb_data" / "GTSRB" / "Final_Training" / "Images"
IMG_SIZE = 64
DEVICE = "cpu"

# id -> name, per the standard GTSRB class ordering. 1-4 share round white
# placard + red border, differing only by digits (the fine-detail case);
# the rest are all shape-distinct (the coarse-cue case).
GTSRB_CLASSES = {
    1: "speed_30", 2: "speed_50", 3: "speed_60", 4: "speed_70",
    12: "priority_road", 13: "yield", 14: "stop", 17: "no_entry",
    18: "general_caution", 38: "keep_right",
}
CLASS_IDS = sorted(GTSRB_CLASSES)
CLASS_NAMES = [GTSRB_CLASSES[c] for c in CLASS_IDS]


def _load_class_records(class_id):
    folder = GTSRB_ROOT / f"{class_id:05d}"
    csv_path = folder / f"GT-{class_id:05d}.csv"
    with open(csv_path, newline="") as f:
        records = list(csv.DictReader(f, delimiter=";"))
    return folder, records


def _track_of(filename):
    return filename.split("_")[0]


def _load_and_crop(folder, record):
    path = folder / record["Filename"]
    x1, y1, x2, y2 = (int(record["Roi.X1"]), int(record["Roi.Y1"]),
                       int(record["Roi.X2"]), int(record["Roi.Y2"]))
    img = Image.open(path).convert("RGB").crop((x1, y1, x2, y2)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_gtsrb_dataset(n_train_per_class=150, n_test_per_class=40, seed=0):
    rnd = random.Random(seed)
    train_imgs, train_labs, test_imgs, test_labs = [], [], [], []
    for label_idx, class_id in enumerate(CLASS_IDS):
        folder, records = _load_class_records(class_id)
        by_track = {}
        for r in records:
            by_track.setdefault(_track_of(r["Filename"]), []).append(r)
        tracks = list(by_track.keys())
        rnd.shuffle(tracks)
        n_test_tracks = max(1, int(0.2 * len(tracks)))
        test_tracks, train_tracks = tracks[:n_test_tracks], tracks[n_test_tracks:]

        def sample_from(track_list, cap, frames_per_track=2):
            pool = []
            for t in track_list:
                frames = by_track[t][:]
                rnd.shuffle(frames)
                pool.extend(frames[:frames_per_track])
            rnd.shuffle(pool)
            return pool[:cap]

        for r in sample_from(train_tracks, n_train_per_class):
            train_imgs.append(_load_and_crop(folder, r))
            train_labs.append(label_idx)
        for r in sample_from(test_tracks, n_test_per_class):
            test_imgs.append(_load_and_crop(folder, r))
            test_labs.append(label_idx)

        print(f"  {GTSRB_CLASSES[class_id]:16s} (id {class_id:2d}): "
              f"{len(tracks)} tracks -> {sum(1 for l in train_labs if l == label_idx)} train, "
              f"{sum(1 for l in test_labs if l == label_idx)} test")

    perm_tr = torch.randperm(len(train_labs))
    perm_te = torch.randperm(len(test_labs))
    return (torch.stack(train_imgs)[perm_tr], torch.tensor(train_labs)[perm_tr],
            torch.stack(test_imgs)[perm_te], torch.tensor(test_labs)[perm_te])


def make_batches(imgs, labs, batch_size=16):
    dummy_mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    return [(imgs[i:i + batch_size], labs[i:i + batch_size],
              dummy_mask.expand(imgs[i:i + batch_size].shape[0], -1, -1, -1))
            for i in range(0, len(labs), batch_size)]


def train_epoch_cls_only(model, dataloader, optimizer, device, pathway_dropout_p=0.0):
    """Same as benchmark.train_baseline_epoch, but with optional pathway
    dropout so it also works for the dual model (no text masks here)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for color_imgs, labels, _ in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, random.random)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)

        r = random.random()
        if r < pathway_dropout_p / 2:
            color_batch = torch.zeros_like(color_batch)
        elif r < pathway_dropout_p:
            gray_batch = torch.zeros_like(gray_batch)

        optimizer.zero_grad()
        logits = model(gray_batch, color_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    if not GTSRB_ROOT.exists():
        raise SystemExit(
            f"{GTSRB_ROOT} not found — download and extract GTSRB first (see module docstring)."
        )

    num_classes = len(CLASS_IDS)
    print(f"Building GTSRB subset ({num_classes} classes: {CLASS_NAMES})...")
    train_imgs, train_labs, test_imgs, test_labs = build_gtsrb_dataset()
    train_loader = make_batches(train_imgs, train_labs)
    test_loader = make_batches(test_imgs, test_labs)
    print(f"Total: {len(train_labs)} train, {len(test_labs)} test (track-disjoint)")

    dual_model = DualPathwayClassifier(num_classes=num_classes, with_text_head=False).to(DEVICE)
    parvo_model = ParvoOnlyClassifier(num_classes=num_classes).to(DEVICE)
    magno_model = MagnoOnlyClassifier(num_classes=num_classes).to(DEVICE)

    opt_dual = Adam(dual_model.parameters(), lr=1e-3)
    opt_parvo = Adam(parvo_model.parameters(), lr=1e-3)
    opt_magno = Adam(magno_model.parameters(), lr=1e-3)

    epochs = 20
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        l_dual = train_epoch_cls_only(dual_model, train_loader, opt_dual, DEVICE, pathway_dropout_p=0.15)
        l_parvo = train_epoch_cls_only(parvo_model, train_loader, opt_parvo, DEVICE)
        l_magno = train_epoch_cls_only(magno_model, train_loader, opt_magno, DEVICE)
        print(f"  epoch {epoch + 1:2d}: dual={l_dual:.3f}  parvo_only={l_parvo:.3f}  magno_only={l_magno:.3f}")

    print("\nCorruption-severity curve (held-out GTSRB test set):")
    results = severity_curve(
        {"dual_pathway": dual_model, "parvo_only": parvo_model, "magno_only": magno_model},
        test_loader, DEVICE,
    )
    print_comparison_table(results)

    print("\nGate weights, clean vs heavily corrupted:")
    gw_clean = avg_gate_weights(dual_model, test_loader, DEVICE, severity=0.0)
    gw_degraded = avg_gate_weights(dual_model, test_loader, DEVICE, severity=1.0)
    print(f"  clean:    magno={gw_clean[0]:.3f} parvo={gw_clean[1]:.3f}")
    print(f"  degraded: magno={gw_degraded[0]:.3f} parvo={gw_degraded[1]:.3f}")

    with open("gtsrb_results.json", "w") as f:
        json.dump({
            "classes": CLASS_NAMES,
            "severity_curve": results,
            "gate_weights": {
                "clean": [gw_clean[0].item(), gw_clean[1].item()],
                "degraded": [gw_degraded[0].item(), gw_degraded[1].item()],
            },
        }, f, indent=2)
    print("\nWrote gtsrb_results.json")


if __name__ == "__main__":
    main()
