"""
Dual-pathway visual classifier inspired by the magnocellular / parvocellular
split in primate vision.

Magno pathway  -> low-fidelity, grayscale, large receptive field, fast.
                  Good for coarse shape/contrast under noise, blur, low light.
Parvo pathway  -> high-fidelity, full color, fine-grained filters, slower.
                  Good for detail (e.g. reading text on a sign) when
                  conditions are decent.

Fusion is a learned gate, not a fixed concat, so the network can shift
weight toward whichever pathway is actually informative for a given input.
Training uses "pathway dropout": each stream is randomly degraded/zeroed
during training so the model can't just always lean on the good one, and
learns robustness for real-world degraded conditions.

Requires: torch, torchvision
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Magnocellular-analog stream: grayscale, downsampled, shallow, wide strides
# --------------------------------------------------------------------------
class MagnoStream(nn.Module):
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


# --------------------------------------------------------------------------
# Parvocellular-analog stream: full color, full resolution, deeper, finer
# --------------------------------------------------------------------------
class ParvoStream(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()

        def block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        # split into a feature trunk (spatial) and pooling, so the text head
        # can tap the spatial map before it gets collapsed to a vector
        self.trunk = nn.Sequential(
            block(in_channels, 32),
            block(32, 64, stride=2),
            block(64, 128, stride=2),
            block(128, 128),
            block(128, feat_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.trunk(x)).flatten(1)

    def forward_spatial(self, x):
        """Returns the spatial feature map (B, feat_dim, H', W') pre-pool,
        for tasks that need location info (e.g. text region detection)."""
        return self.trunk(x)


# --------------------------------------------------------------------------
# Gated fusion: learns per-sample how much to trust each pathway
# --------------------------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, magno_dim, parvo_dim, fused_dim=256):
        super().__init__()
        self.proj_m = nn.Linear(magno_dim, fused_dim)
        self.proj_p = nn.Linear(parvo_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(magno_dim + parvo_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, 2),  # weight per pathway
        )

    def forward(self, m_feat, p_feat):
        gate_logits = self.gate(torch.cat([m_feat, p_feat], dim=1))
        weights = F.softmax(gate_logits, dim=1)  # (B, 2)
        fused = weights[:, 0:1] * self.proj_m(m_feat) + weights[:, 1:2] * self.proj_p(p_feat)
        return fused, weights


# --------------------------------------------------------------------------
# Text region head: parvo-only, since magno's resolution is too low to
# resolve characters. This is a *detector*, not an OCR engine — it predicts
# where text/sign-like regions are, at a spatial resolution matching parvo's
# feature map. Feed the localized crops it produces into a real OCR engine
# (e.g. pytesseract, easyocr, or a CRNN head) for actual character reading.
# --------------------------------------------------------------------------
class TextRegionHead(nn.Module):
    def __init__(self, parvo_feat_dim=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(parvo_feat_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),  # per-pixel text-likelihood logit
        )

    def forward(self, spatial_feat, out_size=None):
        mask_logits = self.net(spatial_feat)
        if out_size is not None:
            mask_logits = F.interpolate(mask_logits, size=out_size, mode="bilinear", align_corners=False)
        return mask_logits  # apply sigmoid outside (BCEWithLogits during training)


def text_regions_from_mask(mask_logits, threshold=0.5, min_area=20):
    """
    Turn a predicted mask (B, 1, H, W) into bounding boxes per image.
    Returns a list (len B) of lists of (x_min, y_min, x_max, y_max) in the
    mask's own pixel coordinates — rescale to your source image size before
    cropping for OCR.
    """
    probs = torch.sigmoid(mask_logits)
    binary = (probs > threshold).squeeze(1)  # (B, H, W)
    all_boxes = []
    for b in range(binary.shape[0]):
        boxes = []
        m = binary[b]
        # simple connected-components via scipy if available, else a
        # coarse fallback that just returns the overall bounding box of
        # all activated pixels (fine for a single sign in frame; swap in
        # cv2.findContours or skimage.measure.label for multi-region cases)
        try:
            from scipy import ndimage
            labeled, n = ndimage.label(m.cpu().numpy())
            for i in range(1, n + 1):
                ys, xs = (labeled == i).nonzero()
                if len(xs) < min_area:
                    continue
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        except ImportError:
            ys, xs = m.nonzero(as_tuple=True)
            if len(xs) >= min_area:
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        all_boxes.append(boxes)
    return all_boxes


# --------------------------------------------------------------------------
# OCR: reads the actual characters out of the regions the text head found.
# Two backends, both optional dependencies — pick whichever you have:
#   "tesseract" -> pytesseract, needs the tesseract-ocr binary installed
#                  on the system (offline, fast, no model download).
#   "easyocr"   -> pure python, downloads a small model on first use,
#                  generally more robust on low-quality/angled text.
# --------------------------------------------------------------------------
def _tensor_crop_to_pil(img_tensor, box, min_height=32):
    """img_tensor: (3, H, W) float in [0,1]. box: (x_min, y_min, x_max, y_max)."""
    from PIL import Image

    x0, y0, x1, y1 = box
    crop = img_tensor[:, y0:y1 + 1, x0:x1 + 1]
    if crop.shape[1] < 2 or crop.shape[2] < 2:
        return None

    # OCR engines do much better on upsampled crops than tiny native-res ones
    h, w = crop.shape[1], crop.shape[2]
    if h < min_height:
        scale = min_height / h
        crop = F.interpolate(
            crop.unsqueeze(0), size=(int(h * scale), int(w * scale)),
            mode="bicubic", align_corners=False,
        ).clamp(0, 1).squeeze(0)

    arr = (crop.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def ocr_read_regions(source_img, boxes, engine="tesseract", lang="eng", psm=7, reader=None):
    """
    source_img: (3, H, W) float tensor in [0,1] — use the highest-resolution
    image you have for this crop (color_highres is fine; a separately
    captured higher-res frame is better if you have one).
    boxes: list of (x_min, y_min, x_max, y_max) in source_img's pixel coords.
    engine: "tesseract" or "easyocr".
    reader: for engine="easyocr", pass an already-constructed easyocr.Reader
    to avoid rebuilding it (and reloading its model) on every call — see
    make_easyocr_reader below.
    Returns a list of strings, one per box ("" if nothing was read).
    """
    texts = []

    if engine == "tesseract":
        try:
            import pytesseract
        except ImportError:
            raise RuntimeError(
                "pytesseract not installed. `pip install pytesseract`, and make sure "
                "the tesseract-ocr system binary is installed (e.g. `apt install tesseract-ocr`)."
            )
        config = f"--psm {psm}"
        for box in boxes:
            pil_img = _tensor_crop_to_pil(source_img, box)
            texts.append(pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()
                         if pil_img is not None else "")

    elif engine == "easyocr":
        if reader is None:
            reader = make_easyocr_reader([lang])
        import numpy as np
        for box in boxes:
            pil_img = _tensor_crop_to_pil(source_img, box)
            if pil_img is None:
                texts.append("")
                continue
            result = reader.readtext(np.array(pil_img))
            texts.append(" ".join(r[1] for r in result) if result else "")

    else:
        raise ValueError(f"Unknown OCR engine: {engine!r}, expected 'tesseract' or 'easyocr'")

    return texts


def make_easyocr_reader(langs=("en",)):
    """Build once, reuse across calls — loading the model per-call is slow."""
    try:
        import easyocr
    except ImportError:
        raise RuntimeError("easyocr not installed. `pip install easyocr`.")
    langs = ["en" if l == "eng" else l for l in langs]
    return easyocr.Reader(langs, gpu=torch.cuda.is_available())


def detect_and_read_text(model, gray_lowres, color_highres, engine="tesseract",
                          lang="eng", reader=None, mask_threshold=0.5, min_area=20):
    """
    Full pipeline in one call: classify, localize likely text/sign regions,
    then OCR each region. Batched — gray_lowres/color_highres are (B, ...).

    Returns (logits, gate_weights, texts_per_sample) where texts_per_sample
    is a list of length B, each entry a list of strings (one per detected
    region in that image, "" for regions OCR couldn't read).

    Uses color_highres itself as the OCR source since it's already full
    resolution; swap in a separately captured higher-res frame per sample
    if you have one and want sharper reads.
    """
    logits, gate_weights, text_mask = model(
        gray_lowres, color_highres, return_gate=True, return_text_mask=True
    )
    boxes_per_sample = text_regions_from_mask(text_mask, threshold=mask_threshold, min_area=min_area)

    if engine == "easyocr" and reader is None:
        reader = make_easyocr_reader([lang])  # build once, reuse across the batch

    texts_per_sample = [
        ocr_read_regions(color_highres[i], boxes, engine=engine, lang=lang, reader=reader)
        for i, boxes in enumerate(boxes_per_sample)
    ]
    return logits, gate_weights, texts_per_sample


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class DualPathwayClassifier(nn.Module):
    def __init__(self, num_classes, magno_dim=128, parvo_dim=256, fused_dim=256,
                 with_text_head=True):
        super().__init__()
        self.magno = MagnoStream(in_channels=1, feat_dim=magno_dim)
        self.parvo = ParvoStream(in_channels=3, feat_dim=parvo_dim)
        self.fusion = GatedFusion(magno_dim, parvo_dim, fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, num_classes),
        )
        self.text_head = TextRegionHead(parvo_feat_dim=parvo_dim) if with_text_head else None

    def forward(self, gray_lowres, color_highres, return_gate=False, return_text_mask=False):
        m_feat = self.magno(gray_lowres)

        if return_text_mask and self.text_head is not None:
            spatial = self.parvo.forward_spatial(color_highres)
            p_feat = self.parvo.pool(spatial).flatten(1)
            text_mask = self.text_head(spatial, out_size=color_highres.shape[-2:])
        else:
            p_feat = self.parvo(color_highres)
            text_mask = None

        fused, gate_weights = self.fusion(m_feat, p_feat)
        logits = self.classifier(fused)

        out = [logits]
        if return_gate:
            out.append(gate_weights)
        if return_text_mask:
            out.append(text_mask)
        return tuple(out) if len(out) > 1 else out[0]


# --------------------------------------------------------------------------
# Input prep: derive both streams from one source image
# --------------------------------------------------------------------------
def make_dual_inputs(color_img, magno_size=32, degrade_p=0.0):
    """
    color_img: tensor (C=3, H, W), float in [0,1], full resolution.
    Returns (gray_lowres, color_highres) ready for the model.

    degrade_p: probability of simulating a bad-condition sample by
    heavily degrading the color stream (blur + noise), forcing the
    network to lean on magno for that sample during training.
    """
    gray = (0.299 * color_img[0] + 0.587 * color_img[1] + 0.114 * color_img[2]).unsqueeze(0)
    gray_lowres = F.interpolate(
        gray.unsqueeze(0), size=(magno_size, magno_size), mode="bilinear", align_corners=False
    ).squeeze(0)

    color_highres = color_img.clone()
    if random.random() < degrade_p:
        noise = torch.randn_like(color_highres) * 0.15
        color_highres = torch.clamp(color_highres + noise, 0, 1)
        color_highres = F.avg_pool2d(color_highres.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)

    return gray_lowres, color_highres


# --------------------------------------------------------------------------
# Minimal training loop skeleton
# --------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device, pathway_dropout_p=0.15,
                     text_mask_weight=0.5):
    """
    dataloader is expected to yield (color_img_batch, label_batch, text_mask_batch),
    where text_mask_batch is (B, 1, H, W) binary ground-truth text/sign masks
    (same H, W as color_img_batch), or None per-sample if unavailable — those
    samples are just skipped for the mask loss and still train the classifier.

    If you don't have text masks yet, pass text_mask_weight=0 and drop the
    mask entries from your dataloader; the classifier head trains fine
    without it, you just won't get the text_head learning anything.

    pathway_dropout_p: probability of zeroing out one entire pathway's
    input per batch, forcing the model to still classify using only the
    other stream. This is what prevents the gate from collapsing onto
    a single dominant pathway.
    """
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in dataloader:
        color_imgs, labels, text_masks = batch
        color_imgs, labels = color_imgs.to(device), labels.to(device)
        has_masks = text_masks is not None and model.text_head is not None
        if has_masks:
            text_masks = text_masks.to(device)

        gray_batch, color_batch = [], []
        for img in color_imgs:
            g, c = make_dual_inputs(img, degrade_p=0.3)
            gray_batch.append(g)
            color_batch.append(c)
        gray_batch = torch.stack(gray_batch).to(device)
        color_batch = torch.stack(color_batch).to(device)

        r = random.random()
        if r < pathway_dropout_p / 2:
            color_batch = torch.zeros_like(color_batch)
            has_masks = False  # no point supervising text on a blanked color stream
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


if __name__ == "__main__":
    # Smoke test with random data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualPathwayClassifier(num_classes=10).to(device)

    gray = torch.rand(4, 1, 32, 32).to(device)
    color = torch.rand(4, 3, 128, 128).to(device)

    logits, gate_weights = model(gray, color, return_gate=True)
    print("logits shape:", logits.shape)
    print("gate weights (magno, parvo) per sample:\n", gate_weights)

    logits, text_mask = model(gray, color, return_text_mask=True)
    print("text mask shape:", text_mask.shape)  # (4, 1, 128, 128), matches color input res
    boxes = text_regions_from_mask(text_mask)
    print("predicted text-region boxes per sample:", boxes)

    try:
        logits, gate_weights, texts = detect_and_read_text(model, gray, color, engine="tesseract")
        print("OCR results per sample:", texts)
    except RuntimeError as e:
        print(f"[OCR demo skipped: {e}]")
