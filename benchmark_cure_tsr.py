"""
Real varied-quality-data version of the severity-curve benchmark, using
CURE-TSR (Challenging Unreal and Real Environments for Traffic Sign
Recognition, Temel et al.) instead of this repo's own synthetic corruption
sweep (benchmark.py) or GTSRB (benchmark_gtsrb.py, which has some natural
variation but no controlled severity levels).

Why CURE-TSR specifically: it ships the SAME real traffic-sign photos
re-rendered under 12 challenge types (decolorization, blur, noise, rain,
snow, haze, exposure, darkening, ...) at 5 severity levels each, plus a
challenge-free version — i.e. real varied-quality data with a built-in,
labeled severity axis, which is exactly what benchmark.py's corrupt_severity
has to fake procedurally. This also makes it a better test bed than GTSRB
for the lateral-connection question (see dual_pathway_classifier.py's
LateralParvoStream / DualPathwayLateralClassifier): does letting magno's
coarse context leak into parvo's trunk mid-forward, on top of the existing
late gated fusion, help more than late fusion alone once the input
degradation is real optical/sensor corruption rather than our own blur+
noise approximation of it?

Not run automatically / not part of the package: CURE-TSR is gated behind a
download-request form, not a direct link (see
https://github.com/olivesgatech/CURE-TSR) and the full release is ~2.2M
images. Request access, download the "Real" sequence archives (this script
only uses sequenceType=01/real, not the unreal/simulator renders), and
extract so that *.bmp files end up somewhere under cure_tsr_data/ next to
this file — any subfolder layout is fine, files are located with rglob and
parsed by filename alone (see CURE-TSR's documented naming convention:
"sequenceType_signType_challengeType_challengeLevel_index.bmp").

Trains on real, challenge-free (challenge=00, level=00) images only — the
realistic framing (see benchmark.py's fusion-necessary experiment for the
same argument): a model is trained on good data and only encounters
degraded input at deployment/test time. Evaluates a severity curve (levels
0-5) on real images for each of a handful of challenge types chosen for
this repo's "low light / low res" theme:
  darkening, exposure, gaussian_blur, decolorization, noise, haze
(the other six CURE-TSR challenges — lens blur, codec error, dirty lens,
rain, shadow, snow — are equally available; this subset just keeps
training/eval time reasonable while covering "too dark", "too bright",
"out of focus", "color signal lost", "sensor noise", and "atmospheric
scattering" as one condition each).

No text-region masks exist in CURE-TSR either (like GTSRB), so all four
models here train classification-only (with_text_head=False for the dual
models).

Run: python benchmark_cure_tsr.py
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam

from dual_pathway_classifier import DualPathwayClassifier, DualPathwayLateralClassifier
from evaluate import MagnoOnlyClassifier, ParvoOnlyClassifier

CURE_TSR_ROOT = Path(__file__).parent / "cure_tsr_data"
IMG_SIZE = 64
DEVICE = "cpu"
MAGNO_SIZE = 16

SIGN_TYPES = {
    1: "speed_limit", 2: "goods_vehicles", 3: "no_overtaking", 4: "no_stopping",
    5: "no_parking", 6: "stop", 7: "bicycle", 8: "hump", 9: "no_left_turn",
    10: "no_right_turn", 11: "priority_to", 12: "no_entry", 13: "yield", 14: "parking",
}
SIGN_IDS = sorted(SIGN_TYPES)
CLASS_NAMES = [SIGN_TYPES[s] for s in SIGN_IDS]

# challenge codes per CURE-TSR's naming convention; 00 is "challenge-free"
# (this script's clean/training condition). The subset actually evaluated
# below is CHALLENGES_TO_TEST, picked for this repo's theme (see docstring).
CHALLENGE_NAMES = {
    0: "challenge_free", 1: "decolorization", 2: "lens_blur", 3: "codec_error",
    4: "darkening", 5: "dirty_lens", 6: "exposure", 7: "gaussian_blur",
    8: "noise", 9: "rain", 10: "shadow", 11: "snow", 12: "haze",
}
CHALLENGES_TO_TEST = (1, 4, 6, 7, 8, 12)  # decolorization, darkening, exposure, gaussian_blur, noise, haze
LEVELS = (0, 1, 2, 3, 4, 5)  # 0 = challenge-free, reused as the shared clean anchor for every curve

_FILENAME_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.bmp$", re.IGNORECASE)


def _parse_filename(path):
    """CURE-TSR naming convention: sequenceType_signType_challengeType_
    challengeLevel_index.bmp. Returns None for anything that doesn't match
    (e.g. readme/license files mixed into the archive)."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    seq, sign, challenge, level, _idx = (int(g) for g in m.groups())
    return seq, sign, challenge, level


def _split_of(path, root):
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    if any("test" in p for p in parts):
        return "test"
    if any("train" in p for p in parts):
        return "train"
    return None


def index_dataset(root, split):
    """Walks root for real (sequenceType==1) *.bmp files and buckets them by
    (sign_type, challenge, level), keeping only files under the release's
    own train or test folders (any directory whose name contains "train" /
    "test", e.g. Real_Train/ and Real_Test/).

    split is required. An earlier version ignored directories and drew
    train and held-out images from one shuffled pool, so "held-out" images
    could be the very photos (or neighbouring video frames) the model was
    trained on, and every degraded test image was a re-render of a frame
    from that same pool. Numbers committed before this fix are optimistic;
    see the README's "Corrections" section."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    buckets = defaultdict(list)
    saw_split_dirs = False
    for path in root.rglob("*.bmp"):
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        seq, sign, challenge, level = parsed
        if seq != 1 or sign not in SIGN_TYPES:
            continue
        s = _split_of(path, root)
        saw_split_dirs |= s is not None
        if s == split:
            buckets[(sign, challenge, level)].append(path)
    if not saw_split_dirs:
        raise SystemExit("No train/test folders found under cure_tsr_data/. Extract CURE-TSR's "
                         "Real_Train and Real_Test archives into separate folders.")
    return buckets


def _load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _sample_split(buckets, challenge, level, n_per_class, seed):
    rnd = random.Random(seed)
    imgs, labs = [], []
    for label_idx, sign_id in enumerate(SIGN_IDS):
        pool = buckets.get((sign_id, challenge, level), [])
        if not pool:
            continue
        pool = pool[:]
        rnd.shuffle(pool)
        for path in pool[:n_per_class]:
            imgs.append(_load_image(path))
            labs.append(label_idx)
    if not imgs:
        return None, None
    imgs, labs = torch.stack(imgs), torch.tensor(labs)
    perm = torch.randperm(len(labs))
    return imgs[perm], labs[perm]


def make_batches(imgs, labs, batch_size=16):
    return [(imgs[i:i + batch_size], labs[i:i + batch_size]) for i in range(0, len(labs), batch_size)]


def _batch_real_inputs(color_imgs, magno_size=MAGNO_SIZE):
    """No synthetic corruption here — the input images are already the
    real degraded photos; magno just gets its usual grayscale+downsample
    derivation of whatever parvo received."""
    gray_batch = []
    for img in color_imgs:
        gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
        gray_batch.append(F.interpolate(
            gray.unsqueeze(0), size=(magno_size, magno_size), mode="bilinear", align_corners=False
        ).squeeze(0))
    return torch.stack(gray_batch), color_imgs


def train_epoch(model, dataloader, optimizer, device, pathway_dropout_p=0.0, dropout_rng=None):
    """dropout_rng: a random.Random instance to draw the pathway-dropout coin
    flip from, instead of the shared global `random` module. Without this,
    every model's dropout draws come from the same process-wide generator,
    so changing one model's training (e.g. a lateral-dropout flag on a
    sibling model trained earlier in the same epoch loop) shifts everyone
    else's coin flips too -- fine for a quick benchmark run, not fine for an
    isolated A/B on one model's own setting. Defaults to the global module
    to keep existing call sites' behavior unchanged."""
    if dropout_rng is None:
        dropout_rng = random
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for color_imgs, labels in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_real_inputs(color_imgs)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)

        r = dropout_rng.random()
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


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for color_imgs, labels in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_real_inputs(color_imgs)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        out = model(gray_batch, color_batch)
        logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else float("nan")


@torch.no_grad()
def avg_gate_weights(model, dataloader, device):
    model.eval()
    weights = []
    for color_imgs, _ in dataloader:
        color_imgs = color_imgs.to(device)
        gray_batch, color_batch = _batch_real_inputs(color_imgs)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        _, gw = model(gray_batch, color_batch, return_gate=True)
        weights.append(gw)
    return torch.cat(weights, dim=0).mean(dim=0)


def main():
    if not CURE_TSR_ROOT.exists():
        raise SystemExit(
            f"{CURE_TSR_ROOT} not found — request and download CURE-TSR first (see module docstring)."
        )

    num_classes = len(SIGN_IDS)
    print(f"Indexing CURE-TSR under {CURE_TSR_ROOT} ({num_classes} sign classes)...")
    train_buckets = index_dataset(CURE_TSR_ROOT, "train")
    buckets = index_dataset(CURE_TSR_ROOT, "test")  # every evaluation set below comes from here
    if not buckets or not train_buckets:
        raise SystemExit("No matching *.bmp files found under cure_tsr_data/ — check the extraction layout.")

    train_imgs, train_labs = _sample_split(train_buckets, challenge=0, level=0, n_per_class=150, seed=0)
    if train_imgs is None:
        raise SystemExit("No challenge-free (challenge=00, level=00) real images found to train on.")
    train_loader = make_batches(train_imgs, train_labs)
    print(f"Train: {len(train_labs)} challenge-free real images")

    # Each model gets its own torch init seed and its own private dropout RNG
    # (see train_epoch's/LateralParvoStream's docstrings) so that re-running
    # this script, or changing one model's hyperparameters, doesn't silently
    # perturb every other model's weight init or per-batch coin flips through
    # a shared global random state.
    torch.manual_seed(1000)
    dual_pathway = DualPathwayClassifier(num_classes=num_classes, with_text_head=False).to(DEVICE)
    torch.manual_seed(1001)
    dual_pathway_lateral = DualPathwayLateralClassifier(
        num_classes=num_classes, with_text_head=False, lateral_dropout_p=0.3,
        lateral_rng_seed=2001).to(DEVICE)
    torch.manual_seed(1002)
    parvo_only = ParvoOnlyClassifier(num_classes=num_classes).to(DEVICE)
    torch.manual_seed(1003)
    magno_only = MagnoOnlyClassifier(num_classes=num_classes).to(DEVICE)

    models = {
        "dual_pathway": dual_pathway,
        "dual_pathway_lateral": dual_pathway_lateral,
        "parvo_only": parvo_only,
        "magno_only": magno_only,
    }
    optimizers = {name: Adam(m.parameters(), lr=1e-3) for name, m in models.items()}
    dropout_p = {"dual_pathway": 0.15, "dual_pathway_lateral": 0.15, "parvo_only": 0.0, "magno_only": 0.0}
    dropout_rngs = {name: random.Random(3000 + i) for i, name in enumerate(models)}

    epochs = 20
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        losses = {name: train_epoch(m, train_loader, optimizers[name], DEVICE, dropout_p[name],
                                     dropout_rng=dropout_rngs[name])
                  for name, m in models.items()}
        print(f"  epoch {epoch + 1:2d}: " + "  ".join(f"{n}={l:.3f}" for n, l in losses.items()))

    print("\nBuilding held-out test sets per (challenge, level) and evaluating...")
    severity_curve = {CHALLENGE_NAMES[c]: {name: {} for name in models} for c in CHALLENGES_TO_TEST}
    clean_test_imgs, clean_test_labs = _sample_split(buckets, challenge=0, level=0, n_per_class=40, seed=1)
    clean_test_loader = make_batches(clean_test_imgs, clean_test_labs) if clean_test_imgs is not None else None

    for challenge in CHALLENGES_TO_TEST:
        cname = CHALLENGE_NAMES[challenge]
        for level in LEVELS:
            if level == 0:
                loader = clean_test_loader  # shared clean anchor, same for every challenge's curve
            else:
                imgs, labs = _sample_split(buckets, challenge=challenge, level=level, n_per_class=40, seed=1)
                loader = make_batches(imgs, labs) if imgs is not None else None
            if loader is None:
                print(f"  [skip] {cname} level {level}: no images found")
                continue
            for name, m in models.items():
                severity_curve[cname][name][level] = evaluate(m, loader, DEVICE)
        print(f"  {cname:16s}: " + "  ".join(
            f"{name}={severity_curve[cname][name].get(5, float('nan')):.3f}@lvl5" for name in models))

    print("\nGate weights, clean vs. heavily degraded (avg across tested challenge types at level 5):")
    gate_weights = {}
    for name in ("dual_pathway", "dual_pathway_lateral"):
        clean = avg_gate_weights(models[name], clean_test_loader, DEVICE)
        degraded_samples = []
        for challenge in CHALLENGES_TO_TEST:
            cname = CHALLENGE_NAMES[challenge]
            imgs, labs = _sample_split(buckets, challenge=challenge, level=5, n_per_class=40, seed=1)
            if imgs is None:
                continue
            loader = make_batches(imgs, labs)
            degraded_samples.append(avg_gate_weights(models[name], loader, DEVICE))
        degraded = torch.stack(degraded_samples).mean(dim=0) if degraded_samples else clean
        gate_weights[name] = {"clean": [clean[0].item(), clean[1].item()],
                               "degraded": [degraded[0].item(), degraded[1].item()]}
        print(f"  {name:20s}: clean magno={clean[0]:.3f} parvo={clean[1]:.3f}  |  "
              f"degraded magno={degraded[0]:.3f} parvo={degraded[1]:.3f}")

    with open("cure_tsr_results.json", "w") as f:
        json.dump({
            "classes": CLASS_NAMES,
            "challenges_tested": [CHALLENGE_NAMES[c] for c in CHALLENGES_TO_TEST],
            "severity_curve": severity_curve,
            "gate_weights": gate_weights,
        }, f, indent=2)
    print("\nWrote cure_tsr_results.json")


if __name__ == "__main__":
    main()
