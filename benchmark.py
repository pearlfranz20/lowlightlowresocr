"""
Reproducible synthetic benchmark for the README's results section.

There's no real labeled dataset in this repo (by design — see README), so
this generates a small procedural "signs" dataset (stop / yield / speed
limit / warning, drawn with PIL, each with a ground-truth text mask) and
trains DualPathwayClassifier + the two ablation baselines briefly. Charts
go to assets/ via plot_results.py.

Corruption: dual_pathway_classifier.make_dual_inputs's built-in degrade
step (a mild single 3x3 blur + std=0.15 noise, applied with probability
degrade_p) turns out too weak to stress this toy dataset — bulk shape and
color survive it easily, so every model saturates at ~100% regardless of
degrade_p (verified). Rather than change that shared function's default
severity (it's used elsewhere, unrelated to this script), this benchmark
defines its own local, continuously-scaled corruption (`corrupt_severity`)
for a legible curve, and trains all three models against a random severity
each step so the dual model's gate actually has a reason to learn. This
also means magno's input resolution (16x16 here, vs the library default of
32x32) differs from evaluate.py's own make_dual_inputs default, so results
aren't directly comparable to evaluate.compare_models() run against the
library defaults — this script's own severity_curve() is used consistently
throughout instead. `evaluate.py`'s real text_detection_metrics is still
exercised directly — see main() below.

This is a toy benchmark meant to demonstrate the architecture's intended
behavior (graceful degradation, gate shifting, text localization), not a
claim about real-world sign/text performance — see the README caveat.

Also runs a second, separate experiment (run_fusion_necessary_experiment):
a task built so that neither pathway alone can do well, unlike the
severity sweep above where fusion only ever turned out to be *sometimes*
helpful. See that function's docstring/comments for the construction.

Run: python benchmark.py
"""

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.optim import Adam

from dual_pathway_classifier import DualPathwayClassifier
from evaluate import (
    MagnoOnlyClassifier,
    ParvoOnlyClassifier,
    print_comparison_table,
    text_detection_metrics,
)

IMG_SIZE = 96
# speed_limit_25 and speed_limit_45 share identical shape/color and differ
# ONLY by their digits — the one pair in this dataset a coarse, shape-only
# view (magno) structurally cannot tell apart, so fusion has a real case to
# earn its keep on top of the shape-diagnostic classes (stop/yield/warning).
CLASSES = ["stop", "yield", "speed_limit_25", "speed_limit_45", "warning"]
DEVICE = "cpu"
SEVERITY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------
# Synthetic "signs" dataset — shapes + text rendered with PIL, plus a
# ground-truth text mask (so we can train/evaluate TextRegionHead too).
# --------------------------------------------------------------------------
def _octagon(cx, cy, r):
    pts = []
    for i in range(8):
        angle = np.pi / 8 + i * np.pi / 4
        pts.append((cx + r * np.sin(angle), cy - r * np.cos(angle)))
    return pts


def _cluttered_background(size):
    base_color = np.random.randint(70, 190, size=3)
    noise = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    arr = (noise * 0.25 + base_color * 0.75).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def render_sample(cls, size=IMG_SIZE):
    img = _cluttered_background(size)
    mask = Image.new("L", (size, size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    cx = size // 2 + random.randint(-6, 6)
    cy = size // 2 + random.randint(-6, 6)
    r = size // 2 - random.randint(8, 16)

    text, text_fill = None, None
    if cls == "stop":
        draw_img.polygon(_octagon(cx, cy, r), fill=(190, 25, 25), outline=(255, 255, 255))
        text, text_fill = "STOP", (255, 255, 255)
    elif cls == "yield":
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw_img.polygon(pts, fill=(250, 245, 235), outline=(190, 25, 25))
        text, text_fill = "YIELD", (190, 25, 25)
    elif cls in ("speed_limit_25", "speed_limit_45"):
        draw_img.rounded_rectangle([cx - r, cy - r, cx + r, cy + r], radius=8,
                                    fill=(250, 250, 245), outline=(20, 20, 20), width=3)
        text = "25" if cls == "speed_limit_25" else "45"
        text_fill = (20, 20, 20)
    elif cls == "warning":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw_img.polygon(pts, fill=(235, 180, 15), outline=(20, 20, 20))

    if text:
        # digits need to be legible enough to actually classify speed_25 vs
        # speed_45 — words (STOP/YIELD) are redundant with shape for
        # classification, so a smaller size (avoiding placard overflow) is
        # fine; they only need to be present for the text-mask/IoU metric.
        font = ImageFont.load_default(size=22 if len(text) <= 2 else 16)
        bbox = draw_img.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]
        draw_img.text((tx, ty), text, fill=text_fill, font=font)
        draw_mask.text((tx, ty), text, fill=255, font=font)

    return img, mask


def _to_tensor(img):
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)


def _mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)


def build_dataset(n_per_class, seed):
    random.seed(seed)
    np.random.seed(seed)
    images, labels, masks = [], [], []
    for idx, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            img, mask = render_sample(cls)
            images.append(_to_tensor(img))
            masks.append(_mask_to_tensor(mask))
            labels.append(idx)
    imgs, labs, msks = torch.stack(images), torch.tensor(labels), torch.stack(masks)
    perm = torch.randperm(len(labs))
    return imgs[perm], labs[perm], msks[perm]


def make_batches(imgs, labs, msks, batch_size=16):
    return [(imgs[i:i + batch_size], labs[i:i + batch_size], msks[i:i + batch_size])
            for i in range(0, len(labs), batch_size)]


# --------------------------------------------------------------------------
# Local, continuously-scaled corruption for this benchmark's own curve —
# see module docstring for why the shared make_dual_inputs degrade step
# isn't used for training/evaluating classification here. Gray/magno input
# is still derived from the clean image, same as make_dual_inputs, so
# magno's accuracy is structurally unaffected by severity — same guarantee
# the real pipeline gives.
# --------------------------------------------------------------------------
def corrupt_severity(color_img, severity):
    """Blur first (destroys fine detail — this is what a low-res/low-light
    capture actually loses), THEN add noise on top of the blurred result.
    Adding noise before blurring is a no-op at any real severity: averaging
    over the kernel launders iid noise back out almost entirely."""
    if severity <= 0:
        return color_img.clone()
    kernel = 3 + 2 * int(round(severity * 5))  # odd, 3..13
    noise_std = 0.10 + 0.45 * severity
    img = F.avg_pool2d(color_img.unsqueeze(0), kernel, stride=1, padding=kernel // 2).squeeze(0)
    img = torch.clamp(img + torch.randn_like(img) * noise_std, 0, 1)
    return img


def make_inputs_severity(color_img, severity, magno_size=16):
    gray = (0.299 * color_img[0] + 0.587 * color_img[1] + 0.114 * color_img[2]).unsqueeze(0)
    gray_lowres = F.interpolate(
        gray.unsqueeze(0), size=(magno_size, magno_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    color_highres = corrupt_severity(color_img, severity)
    return gray_lowres, color_highres


def _batch_inputs(color_imgs, severity_fn):
    gray_batch, color_batch = [], []
    for img in color_imgs:
        g, c = make_inputs_severity(img, severity_fn())
        gray_batch.append(g)
        color_batch.append(c)
    return torch.stack(gray_batch), torch.stack(color_batch)


# --------------------------------------------------------------------------
# Training — random severity per sample each step, so parvo sometimes gets
# genuinely unreadable input and the dual model's gate has a reason to
# learn to lean on magno. Mirrors train_one_epoch's pathway-dropout logic
# for the dual model; baselines only have one pathway so they just train
# straight through.
# --------------------------------------------------------------------------
def train_dual_epoch(model, dataloader, optimizer, device, pathway_dropout_p=0.15, text_mask_weight=2.0):
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    # text pixels are a small minority of each image; without pos_weight the
    # mask head trivially collapses to "no text anywhere" (cheap low BCE loss).
    mask_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(15.0, device=device))
    total_loss = 0.0
    for color_imgs, labels, text_masks in dataloader:
        color_imgs, labels, text_masks = color_imgs.to(device), labels.to(device), text_masks.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, random.random)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)

        has_masks = True
        r = random.random()
        if r < pathway_dropout_p / 2:
            color_batch = torch.zeros_like(color_batch)
            has_masks = False
        elif r < pathway_dropout_p:
            gray_batch = torch.zeros_like(gray_batch)

        optimizer.zero_grad()
        if has_masks:
            logits, pred_mask = model(gray_batch, color_batch, return_text_mask=True)
            loss = cls_criterion(logits, labels) + text_mask_weight * mask_criterion(pred_mask, text_masks)
        else:
            logits = model(gray_batch, color_batch)
            loss = cls_criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_baseline_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for color_imgs, labels, _ in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, random.random)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)

        optimizer.zero_grad()
        logits = model(gray_batch, color_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_at_severity(model, dataloader, device, severity):
    model.eval()
    correct, total = 0, 0
    for color_imgs, labels, _ in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, lambda: severity)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        out = model(gray_batch, color_batch)
        logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else float("nan")


def severity_curve(models: dict, dataloader, device, severity_levels=SEVERITY_LEVELS):
    return {name: {s: evaluate_at_severity(m, dataloader, device, s) for s in severity_levels}
            for name, m in models.items()}


@torch.no_grad()
def avg_gate_weights(model, dataloader, device, severity):
    model.eval()
    weights = []
    for color_imgs, _, _ in dataloader:
        color_imgs = color_imgs.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, lambda: severity)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        _, gw = model(gray_batch, color_batch, return_gate=True)
        weights.append(gw)
    return torch.cat(weights, dim=0).mean(dim=0)  # (magno_w, parvo_w)


@torch.no_grad()
def avg_text_metrics(model, dataloader, device, severity):
    model.eval()
    all_metrics = []
    for color_imgs, _, masks in dataloader:
        color_imgs, masks = color_imgs.to(device), masks.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, lambda: severity)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        _, pred_mask = model(gray_batch, color_batch, return_text_mask=True)
        all_metrics.append(text_detection_metrics(pred_mask, masks))
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}


# --------------------------------------------------------------------------
# Fusion-necessary task: unlike the severity sweep above (which showed
# fusion is *sometimes helpful* under corruption but never *required* —
# every task tried always turned out solvable by whichever single pathway
# was stronger, per the GTSRB README section on this), this constructs a
# task where NEITHER pathway alone can do well, by design.
#
# Every sample gets exactly one whole pathway zeroed, chosen independently
# per sample (~50/50) — not blur, not partial noise, a hard per-sample
# either/or ("half your shots come from a color camera with a dead color
# sensor, half from a mono-only camera"). A single-pathway model can only
# ever succeed on the ~half of samples where ITS pathway happens to be the
# survivor; only a model that can detect which pathway is live per sample
# and route to it can do well across the whole set. This is exactly what
# GatedFusion is for (a per-sample, not global, weighting) — this task is
# built specifically to need that.
# --------------------------------------------------------------------------
def zero_one_pathway_batch(gray_batch, color_batch):
    keep_gray = torch.rand(gray_batch.shape[0], device=gray_batch.device) < 0.5
    gray_out = gray_batch * keep_gray.view(-1, 1, 1, 1)
    color_out = color_batch * (~keep_gray).view(-1, 1, 1, 1)
    return gray_out, color_out


def train_fusion_necessary_epoch(model, dataloader, optimizer, device, text_mask_weight=2.0):
    """Trains DualPathwayClassifier under the always-one-pathway-missing
    regime. Ordinary pathway dropout (a *fraction* of batches) isn't
    frequent enough exposure for the gate to learn a reliable per-sample
    detect-and-route policy — this task needs it on every sample."""
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(15.0, device=device))
    total_loss = 0.0
    for color_imgs, labels, text_masks in dataloader:
        color_imgs, labels, text_masks = color_imgs.to(device), labels.to(device), text_masks.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, lambda: 0.0)  # clean, pre-zeroing
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
        gray_batch, color_batch = zero_one_pathway_batch(gray_batch, color_batch)

        optimizer.zero_grad()
        logits, pred_mask = model(gray_batch, color_batch, return_text_mask=True)
        loss = cls_criterion(logits, labels) + text_mask_weight * mask_criterion(pred_mask, text_masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_baseline_clean_epoch(model, dataloader, optimizer, device):
    """Single-pathway baselines for the fusion-necessary task are trained
    on clean data only, not exposed to zeroed input during training —
    realistic framing: a single-sensor system is trained on good data for
    its one sensor, then just happens to lose signal sometimes at
    deployment/test time. (If you knew in advance when your one sensor
    would fail, you'd just use a different sensor.)"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for color_imgs, labels, _ in dataloader:
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        gray_batch, color_batch = _batch_inputs(color_imgs, lambda: 0.0)
        gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)

        optimizer.zero_grad()
        logits = model(gray_batch, color_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_fusion_necessary(model, dataloader, device, n_trials=5):
    """Accuracy under the same one-pathway-always-missing regime. n_trials
    repeats the random per-sample zeroing draw and averages, since a single
    draw is noisy (~half the test set is arbitrary per draw)."""
    model.eval()
    accs = []
    for _ in range(n_trials):
        correct, total = 0, 0
        for color_imgs, labels, _ in dataloader:
            color_imgs, labels = color_imgs.to(device), labels.to(device)
            gray_batch, color_batch = _batch_inputs(color_imgs, lambda: 0.0)
            gray_batch, color_batch = gray_batch.to(device), color_batch.to(device)
            gray_batch, color_batch = zero_one_pathway_batch(gray_batch, color_batch)
            out = model(gray_batch, color_batch)
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        accs.append(correct / total if total else float("nan"))
    return sum(accs) / len(accs)


def run_fusion_necessary_experiment(train_loader, test_loader, num_classes, epochs=22):
    print(f"\n{'=' * 70}\nFusion-necessary task: every sample has exactly one pathway zeroed\n{'=' * 70}")
    dual_model = DualPathwayClassifier(num_classes=num_classes, with_text_head=True).to(DEVICE)
    parvo_model = ParvoOnlyClassifier(num_classes=num_classes).to(DEVICE)
    magno_model = MagnoOnlyClassifier(num_classes=num_classes).to(DEVICE)

    opt_dual = Adam(dual_model.parameters(), lr=1e-3)
    opt_parvo = Adam(parvo_model.parameters(), lr=1e-3)
    opt_magno = Adam(magno_model.parameters(), lr=1e-3)

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        l_dual = train_fusion_necessary_epoch(dual_model, train_loader, opt_dual, DEVICE)
        l_parvo = train_baseline_clean_epoch(parvo_model, train_loader, opt_parvo, DEVICE)
        l_magno = train_baseline_clean_epoch(magno_model, train_loader, opt_magno, DEVICE)
        print(f"  epoch {epoch + 1:2d}: dual={l_dual:.3f}  parvo_only={l_parvo:.3f}  magno_only={l_magno:.3f}")

    print("\nAccuracy with one pathway always missing (averaged over 5 random draws):")
    results = {
        "dual_pathway": evaluate_fusion_necessary(dual_model, test_loader, DEVICE),
        "parvo_only": evaluate_fusion_necessary(parvo_model, test_loader, DEVICE),
        "magno_only": evaluate_fusion_necessary(magno_model, test_loader, DEVICE),
    }
    for name, acc in results.items():
        print(f"  {name:14s}: {acc:.4f}")

    with open("fusion_necessary_results.json", "w") as f:
        json.dump({"num_classes": num_classes, "epochs": epochs, "accuracy": results}, f, indent=2)
    print("\nWrote fusion_necessary_results.json")
    return results


def main():
    num_classes = len(CLASSES)
    print("Building synthetic dataset...")
    train_imgs, train_labs, train_msks = build_dataset(n_per_class=150, seed=0)
    test_imgs, test_labs, test_msks = build_dataset(n_per_class=50, seed=1)
    train_loader = make_batches(train_imgs, train_labs, train_msks)
    test_loader = make_batches(test_imgs, test_labs, test_msks)

    dual_model = DualPathwayClassifier(num_classes=num_classes, with_text_head=True).to(DEVICE)
    parvo_model = ParvoOnlyClassifier(num_classes=num_classes).to(DEVICE)
    magno_model = MagnoOnlyClassifier(num_classes=num_classes).to(DEVICE)

    opt_dual = Adam(dual_model.parameters(), lr=1e-3)
    opt_parvo = Adam(parvo_model.parameters(), lr=1e-3)
    opt_magno = Adam(magno_model.parameters(), lr=1e-3)

    epochs = 22
    print(f"Training for {epochs} epochs on {len(train_labs)} synthetic samples "
          f"(random corruption severity each step)...")
    for epoch in range(epochs):
        l_dual = train_dual_epoch(dual_model, train_loader, opt_dual, DEVICE)
        l_parvo = train_baseline_epoch(parvo_model, train_loader, opt_parvo, DEVICE)
        l_magno = train_baseline_epoch(magno_model, train_loader, opt_magno, DEVICE)
        print(f"  epoch {epoch + 1:2d}: dual={l_dual:.3f}  parvo_only={l_parvo:.3f}  magno_only={l_magno:.3f}")

    print("\nCorruption-severity curve (held-out test set):")
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

    print("\nText detection quality (evaluate.text_detection_metrics), clean vs heavily corrupted:")
    tm_clean = avg_text_metrics(dual_model, test_loader, DEVICE, severity=0.0)
    tm_degraded = avg_text_metrics(dual_model, test_loader, DEVICE, severity=1.0)
    print(f"  clean:    {tm_clean}")
    print(f"  degraded: {tm_degraded}")

    with open("benchmark_results.json", "w") as f:
        json.dump({
            "severity_curve": results,
            "gate_weights": {
                "clean": [gw_clean[0].item(), gw_clean[1].item()],
                "degraded": [gw_degraded[0].item(), gw_degraded[1].item()],
            },
            "text_metrics": {"clean": tm_clean, "degraded": tm_degraded},
        }, f, indent=2)
    print("\nWrote benchmark_results.json")

    run_fusion_necessary_experiment(train_loader, test_loader, num_classes)


if __name__ == "__main__":
    main()
