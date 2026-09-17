"""
Evaluation harness for the dual-pathway classifier.

The model's whole premise is graceful degradation under bad conditions, so
the metrics here are built around comparing performance ACROSS degradation
levels and against simpler baselines, not just a single clean-condition
accuracy number.

Three things this module answers:
  1. Does fusion actually beat using either pathway alone? (ablation)
  2. Does accuracy degrade more gracefully than the baselines as conditions
     worsen? (the actual point of the architecture)
  3. Does localizing text before OCR beat running OCR on the whole image?
     (the specific question of "why not just OCR the raw image")

Requires: torch. OCR comparison additionally needs pytesseract/easyocr,
same as dual_pathway_classifier.py.
"""

import torch
import torch.nn as nn

from dual_pathway_classifier import (
    MagnoStream,
    ParvoStream,
    make_dual_inputs,
    ocr_read_regions,
    detect_and_read_text,
)


# --------------------------------------------------------------------------
# Ablation baselines: single-pathway classifiers with the same interface
# as DualPathwayClassifier, so they drop into the same eval loop.
# --------------------------------------------------------------------------
class MagnoOnlyClassifier(nn.Module):
    """Ablation: what happens if you throw away the parvo stream entirely."""

    def __init__(self, num_classes, magno_dim=128):
        super().__init__()
        self.magno = MagnoStream(in_channels=1, feat_dim=magno_dim)
        self.classifier = nn.Sequential(
            nn.Linear(magno_dim, magno_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(magno_dim // 2, num_classes),
        )

    def forward(self, gray_lowres, color_highres=None):
        return self.classifier(self.magno(gray_lowres))


class ParvoOnlyClassifier(nn.Module):
    """Ablation: a conventional color-CNN baseline with no magno stream at all —
    this is the 'just use a normal classifier' comparison point."""

    def __init__(self, num_classes, parvo_dim=256):
        super().__init__()
        self.parvo = ParvoStream(in_channels=3, feat_dim=parvo_dim)
        self.classifier = nn.Sequential(
            nn.Linear(parvo_dim, parvo_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(parvo_dim // 2, num_classes),
        )

    def forward(self, gray_lowres=None, color_highres=None):
        return self.classifier(self.parvo(color_highres))


# --------------------------------------------------------------------------
# Classification: accuracy at a given degradation level
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_classification(model, dataloader, device, degrade_p=0.0):
    """
    degrade_p is passed straight to make_dual_inputs: 0.0 = clean color
    input, 1.0 = every sample gets its color stream degraded (blur+noise).
    Use intermediate values, or run at several values, to trace out a
    degradation curve rather than a single number.
    """
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        color_imgs, labels = batch[0], batch[1]
        color_imgs, labels = color_imgs.to(device), labels.to(device)

        gray_batch, color_batch = [], []
        for img in color_imgs:
            g, c = make_dual_inputs(img, degrade_p=degrade_p)
            gray_batch.append(g)
            color_batch.append(c)
        gray_batch = torch.stack(gray_batch).to(device)
        color_batch = torch.stack(color_batch).to(device)

        out = model(gray_batch, color_batch)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total else float("nan")


def degradation_curve(model, dataloader, device, degrade_levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Accuracy at each degradation level — the shape of this curve matters
    more than any single point on it. A model that's robust should stay
    flatter as degrade_p rises; a color-only model should fall off a cliff."""
    return {p: evaluate_classification(model, dataloader, device, degrade_p=p) for p in degrade_levels}


def compare_models(models: dict, dataloader, device, degrade_levels=(0.0, 0.5, 1.0)):
    """
    models: {"name": model_instance, ...} — e.g.
      {"dual_pathway": dual_model, "parvo_only": ParvoOnlyClassifier(...),
       "magno_only": MagnoOnlyClassifier(...)}
    Returns {"name": {degrade_p: accuracy, ...}, ...} for a side-by-side table.
    """
    return {name: degradation_curve(m, dataloader, device, degrade_levels) for name, m in models.items()}


def print_comparison_table(results: dict):
    levels = sorted(next(iter(results.values())).keys())
    header = "model".ljust(18) + "".join(f"degrade={p:<10.2f}" for p in levels)
    print(header)
    print("-" * len(header))
    for name, accs in results.items():
        row = name.ljust(18) + "".join(f"{accs[p]:<18.3f}" for p in levels)
        print(row)


# --------------------------------------------------------------------------
# Text detection: pixel-level IoU / precision / recall / F1 against
# ground-truth masks
# --------------------------------------------------------------------------
@torch.no_grad()
def text_detection_metrics(pred_mask_logits, gt_mask, threshold=0.5):
    """
    pred_mask_logits: (B, 1, H, W) raw logits from TextRegionHead.
    gt_mask: (B, 1, H, W) binary ground-truth text-region mask, same size.
    Returns dict of mean IoU/precision/recall/F1 across the batch.
    """
    probs = torch.sigmoid(pred_mask_logits)
    pred = (probs > threshold).float()
    gt = gt_mask.float()

    intersection = (pred * gt).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    gt_sum = gt.sum(dim=(1, 2, 3))
    union = pred_sum + gt_sum - intersection

    iou = (intersection / union.clamp(min=1e-6)).mean().item()
    precision = (intersection / pred_sum.clamp(min=1e-6)).mean().item()
    recall = (intersection / gt_sum.clamp(min=1e-6)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------
# OCR accuracy: character/word error rate, and the raw-OCR-vs-pipeline
# comparison you specifically asked about
# --------------------------------------------------------------------------
def _edit_distance(seq_a, seq_b):
    """Generic Levenshtein distance, works on strings or lists of words."""
    n, m = len(seq_a), len(seq_b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if seq_a[i - 1] == seq_b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def character_error_rate(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(pred, gt) / len(gt)


def word_error_rate(pred: str, gt: str) -> float:
    pred_words, gt_words = pred.split(), gt.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit_distance(pred_words, gt_words) / len(gt_words)


def evaluate_ocr_pipeline(model, samples, engine="tesseract", lang="eng", degrade_p=0.0, reader=None):
    """
    samples: list of (color_img_tensor, ground_truth_text) pairs. Each
    color_img_tensor is (3, H, W) float in [0, 1], ideally one sign/text
    region roughly filling the frame, paired with its correct transcription.

    Compares, per sample:
      raw_ocr          -> OCR run directly on the whole image, no localization.
      detect_then_ocr  -> this repo's pipeline: TextRegionHead finds the
                           region(s) first, then OCR reads just the crop(s).

    This is the direct answer to "why not just OCR the image": raw_ocr is
    that baseline. Expect the gap between the two to widen as degrade_p
    rises — that's the localization step earning its keep, since a tight
    crop is less confused by background clutter than the full frame.

    Returns {"raw_ocr": {"cer":.., "wer":..}, "detect_then_ocr": {...}}.
    """
    raw_cers, raw_wers, pipe_cers, pipe_wers = [], [], [], []
    model.eval()

    for color_img, gt_text in samples:
        gray, color = make_dual_inputs(color_img, degrade_p=degrade_p)
        gray_b, color_b = gray.unsqueeze(0), color.unsqueeze(0)

        full_box = (0, 0, color.shape[2] - 1, color.shape[1] - 1)
        raw_text = ocr_read_regions(color, [full_box], engine=engine, lang=lang, reader=reader)[0]
        raw_cers.append(character_error_rate(raw_text, gt_text))
        raw_wers.append(word_error_rate(raw_text, gt_text))

        with torch.no_grad():
            _, _, texts = detect_and_read_text(model, gray_b, color_b, engine=engine, lang=lang, reader=reader)
        pipe_text = " ".join(texts[0]) if texts[0] else ""
        pipe_cers.append(character_error_rate(pipe_text, gt_text))
        pipe_wers.append(word_error_rate(pipe_text, gt_text))

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "raw_ocr": {"cer": avg(raw_cers), "wer": avg(raw_wers)},
        "detect_then_ocr": {"cer": avg(pipe_cers), "wer": avg(pipe_wers)},
    }


if __name__ == "__main__":
    from dual_pathway_classifier import DualPathwayClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10

    dual_model = DualPathwayClassifier(num_classes=num_classes, with_text_head=False).to(device)
    parvo_model = ParvoOnlyClassifier(num_classes=num_classes).to(device)
    magno_model = MagnoOnlyClassifier(num_classes=num_classes).to(device)

    # Fake dataset for a smoke test — replace with a real DataLoader.
    fake_images = torch.rand(16, 3, 128, 128)
    fake_labels = torch.randint(0, num_classes, (16,))
    fake_loader = [(fake_images[i:i + 4], fake_labels[i:i + 4]) for i in range(0, 16, 4)]

    results = compare_models(
        {"dual_pathway": dual_model, "parvo_only": parvo_model, "magno_only": magno_model},
        fake_loader,
        device,
    )
    print_comparison_table(results)
    print(
        "\n(Untrained random weights above — this just proves the harness runs. "
        "Train each model first for numbers that mean anything.)"
    )
