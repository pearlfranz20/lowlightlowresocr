# Dual-Pathway Visual Classifier

A visual classifier inspired by the magnocellular / parvocellular split in
primate vision — one fast, coarse, low-fidelity pathway and one slow,
detailed, high-fidelity pathway, fused by a learned gate. Built with signs
and text-in-the-wild in mind: conditions where lighting, blur, or distance
can wreck fine detail but coarse shape still gets through.

## Why two pathways

| | Magno pathway | Parvo pathway |
|---|---|---|
| Input | grayscale, downsampled | full color, full resolution |
| Depth | shallow, wide receptive field | deeper, fine-grained filters |
| Speed | fast | slower |
| Good at | coarse shape/contrast, robust to noise & blur | fine detail, reading text, color cues |
| Analogous to | motion/contrast-sensitive magnocellular cells | detail/color-sensitive parvocellular cells |

Rather than picking one input representation and hoping it generalizes, the
model always sees both and learns *when to trust which one*.

## Architecture

```
gray_lowres ──▶ MagnoStream ──┐
                               ├──▶ GatedFusion ──▶ classifier ──▶ logits
color_highres ─▶ ParvoStream ─┘
                     │
                     └─▶ TextRegionHead ──▶ text mask ──▶ OCR
```

- **`MagnoStream`** — grayscale in, large strides, small feature dim. Cheap and fast.
- **`ParvoStream`** — color in, deeper conv stack. Exposes its pre-pool spatial
  feature map (`forward_spatial`) for the text head to use.
- **`GatedFusion`** — a small network looks at both feature vectors and predicts
  a per-sample weighting between them, rather than a fixed concat. Log these
  weights at inference time to see which pathway the model is actually relying
  on for a given input.
- **`TextRegionHead`** — parvo-only (magno's resolution can't resolve
  characters). Predicts a per-pixel text/sign-likelihood mask, which
  `text_regions_from_mask` turns into bounding boxes.
- **OCR (`ocr_read_regions`, `detect_and_read_text`)** — crops each detected
  region and reads it with `pytesseract` or `easyocr`. This is the only part
  that reads characters; the network itself only detects *where* text is.

## Training strategy

Two things keep the model honest:

1. **Random pathway degradation** (`make_dual_inputs(..., degrade_p=...)`) —
   simulates bad conditions by blurring/adding noise to the color stream for
   a fraction of samples, so magno has to carry those cases.
2. **Pathway dropout** (`train_one_epoch(..., pathway_dropout_p=...)`) —
   randomly zeroes out one entire pathway per batch, so the classifier can't
   just learn to always lean on parvo and ignore magno (or vice versa).

Without both of these, the gate tends to collapse onto whichever pathway is
higher-fidelity on average and stops learning to use the other one when it
actually needs to.

## Quickstart

```python
import torch
from dual_pathway_classifier import DualPathwayClassifier, make_dual_inputs, detect_and_read_text

model = DualPathwayClassifier(num_classes=10)

# from a single full-res color image tensor (3, H, W) in [0, 1]
gray, color = make_dual_inputs(my_image)
gray, color = gray.unsqueeze(0), color.unsqueeze(0)  # add batch dim

logits, gate_weights = model(gray, color, return_gate=True)

# classify + find + read text regions in one call
logits, gate_weights, texts = detect_and_read_text(model, gray, color, engine="tesseract")
```

Training loop:

```python
from torch.optim import Adam
from dual_pathway_classifier import train_one_epoch

optimizer = Adam(model.parameters(), lr=1e-3)
# dataloader yields (color_img_batch, label_batch, text_mask_batch_or_None)
loss = train_one_epoch(model, dataloader, optimizer, device="cpu")
```

## Requirements

- `torch`
- `scipy` (optional — enables proper multi-region connected-component
  detection in `text_regions_from_mask`; falls back to a single bounding box
  over all activated pixels without it)
- `pytesseract` + the `tesseract-ocr` system binary, **or** `easyocr`
  (optional — only needed for `ocr_read_regions` / `detect_and_read_text`)

```bash
pip install torch scipy
pip install pytesseract        # + apt install tesseract-ocr (or your OS equivalent)
# or
pip install easyocr
```

## Evaluation

`evaluate.py` is built around the fact that this architecture's whole point
is graceful degradation — so the metrics compare **across degradation
levels and against simpler baselines**, not just one clean-condition number.

**Ablation baselines** — `ParvoOnlyClassifier` and `MagnoOnlyClassifier`
drop into the same eval loop as the dual model, so you can directly check
whether fusion is earning its complexity over "just a normal color CNN" or
"just a lightweight grayscale CNN."

**Degradation curves**, not single numbers:

```python
from evaluate import compare_models, print_comparison_table

results = compare_models(
    {"dual_pathway": dual_model, "parvo_only": parvo_baseline, "magno_only": magno_baseline},
    test_loader, device, degrade_levels=(0.0, 0.25, 0.5, 0.75, 1.0),
)
print_comparison_table(results)
```

The shape of the curve is the point: `parvo_only` should fall off sharply as
`degrade_p` rises, `magno_only` should stay flat but low overall, and
`dual_pathway` should stay both high and flat if the gate is doing its job.

**Text detection** — pixel-level IoU/precision/recall/F1 for the region
head against ground-truth masks (`text_detection_metrics`).

**OCR: this pipeline vs. straight OCR on the raw image** — the specific
comparison worth running: does localizing before OCR actually help?

```python
from evaluate import evaluate_ocr_pipeline

# samples: list of (color_img_tensor, ground_truth_text) pairs
results = evaluate_ocr_pipeline(model, samples, engine="tesseract", degrade_p=0.5)
# {"raw_ocr": {"cer": ..., "wer": ...}, "detect_then_ocr": {"cer": ..., "wer": ...}}
```

Character/word error rate (CER/WER) are the standard OCR metrics — lower is
better, 0 is a perfect transcription. Expect `raw_ocr` and `detect_then_ocr`
to be close on clean, uncluttered images and to diverge as `degrade_p` rises
or background clutter increases — that gap is what localization buys you.
Run it at a few `degrade_p` values rather than just one to see the trend.

### Public benchmarks worth testing against

For a more rigorous comparison than your own held-out set:

- **[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)** (German Traffic
  Sign Recognition Benchmark) — classification, has real varied lighting/blur.
- **[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)** / **COCO-Text** — scene text
  detection, standard for evaluating region proposals like `TextRegionHead`.
- **[SVHN](http://ufldl.stanford.edu/housenumbers/)** — house-number digits
  in natural scenes, a reasonable proxy for "read small text in the wild."
- Off-the-shelf detectors like **EAST** or **CRAFT** are the standard
  comparison points for text localization specifically, if you want to
  benchmark `TextRegionHead` against dedicated text-detection models rather
  than just against raw-OCR.



- Not a full OCR engine — `TextRegionHead` finds *where* text is, it doesn't
  read it. Reading is delegated to `pytesseract`/`easyocr`.
- Not pretrained — this is architecture + training scaffolding. You'll need
  your own labeled dataset (images, class labels, and optionally text-region
  masks) to train it.
- Box extraction without `scipy` is a coarse single-box fallback — fine for
  one sign in frame, not for scenes with multiple separated text regions.

## File overview

| File | Contents |
|---|---|
| `dual_pathway_classifier.py` | Full model, fusion, text head, OCR integration, training loop |
| `evaluate.py` | Baselines, degradation-curve metrics, text-detection metrics, OCR comparison |
