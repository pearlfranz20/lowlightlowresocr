"""
Optional decision layer on top of DualPathwayClassifier, using Jev
(typesafe.ai's System One model) to turn raw model outputs into a single
calibrated, type-safe downstream decision.

The classifier already tells you *what* it thinks (logits), *how* it got
there (gate_weights — how much it leaned on magno vs. parvo), and *what it
read* (OCR text per detected region). None of that is, by itself, a
decision an automation pipeline can safely act on: logits aren't calibrated
probabilities, gate weights don't say whether to trust the call, and OCR
output has no confidence signal at all.

This module packs those three signals into a `state` description and asks
Jev one combined question: is the classification trustworthy, is the OCR
reading trustworthy, and what should downstream automation actually do.

Requires: typesafe-sdk (`pip install typesafe-sdk`), TYPESAFE_API_KEY set.
`decide()`/`decide_batch()` only need typesafe-sdk; `detect_read_and_decide()`
additionally needs torch + the classifier, imported lazily so the rest of
this module works without them installed.

Batch decisions are independent per sample, so `decide_batch` fires them
concurrently (via `AsyncTypeSafeClient`) instead of one-at-a-time — a batch
of B samples takes roughly one call's latency, not B of them.
"""


def _get_client():
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        raise RuntimeError(
            "typesafe-sdk not installed. `pip install typesafe-sdk`, and set "
            "the TYPESAFE_API_KEY environment variable."
        )
    return TypeSafeClient()


def _get_async_client():
    try:
        from typesafe_sdk import AsyncTypeSafeClient
    except ImportError:
        raise RuntimeError(
            "typesafe-sdk not installed. `pip install typesafe-sdk`, and set "
            "the TYPESAFE_API_KEY environment variable."
        )
    return AsyncTypeSafeClient()


def build_state(class_name, class_prob, gate_weights, ocr_texts):
    """
    class_name: predicted class label (str or int).
    class_prob: softmax probability of the predicted class (float).
    gate_weights: (magno_weight, parvo_weight) floats, sum to 1.
    ocr_texts: list of strings read from detected regions (may be empty).
    """
    magno_w, parvo_w = gate_weights
    lines = [
        f"Vision model predicted class {class_name!r} with probability {class_prob:.2f}.",
        f"Pathway trust weighting: magno (coarse/robust) = {magno_w:.2f}, "
        f"parvo (fine detail) = {parvo_w:.2f}.",
    ]
    read_texts = [t for t in ocr_texts if t]
    if read_texts:
        lines.append("OCR read the following text from detected regions: "
                      + "; ".join(repr(t) for t in read_texts))
    else:
        lines.append("No text was read from any detected region "
                      "(either none were detected, or OCR returned nothing).")
    return "\n".join(lines)


def _build_questions(class_name, class_prob, gate_weights, ocr_texts):
    """Shared by decide() and adecide(): builds the (state, questions) pair
    for one combined Jev call — no I/O, so both the sync and async paths
    stay in exact sync."""
    from typesafe_sdk import Choice, Noul

    state = build_state(class_name, class_prob, gate_weights, ocr_texts)
    has_ocr = any(ocr_texts)

    questions = {
        "trust_classification": Noul(
            instructions="The predicted class is correct and reliable enough "
                         "for downstream automation to act on, given the "
                         "model's confidence and pathway weighting described above.",
        ),
        "action": Choice(
            instructions="What should downstream automation do with this observation?",
            criteria={
                "act": "Confidence and pathway weighting are high enough to "
                       "act on this observation directly.",
                "flag_for_review": "Signal is mixed or borderline — a human "
                                   "or a slower system should double-check before acting.",
                "discard": "Confidence is too low, or the pathways disagree "
                           "too strongly, to be worth acting on at all.",
            },
        ),
    }
    if has_ocr:
        questions["ocr_reliable"] = Noul(
            instructions="The OCR text read from the detected regions above "
                         "is an accurate transcription, trustworthy enough "
                         "for downstream use.",
        )
    return state, questions


def decide(class_name, class_prob, gate_weights, ocr_texts, client=None):
    """
    One combined Jev call: is the classification trustworthy, is the OCR
    reading (if any) trustworthy, and what should downstream automation do.

    Returns the raw typesafe_sdk response object — access via
    response.answers["trust_classification"].noul, etc.
    """
    if client is None:
        client = _get_client()
    state, questions = _build_questions(class_name, class_prob, gate_weights, ocr_texts)
    return client.system_one(state=state, questions=questions)


async def adecide(class_name, class_prob, gate_weights, ocr_texts, client=None):
    """Async version of decide() — same call, non-blocking, so many of these
    can run concurrently under asyncio.gather instead of one at a time."""
    if client is None:
        client = _get_async_client()
    state, questions = _build_questions(class_name, class_prob, gate_weights, ocr_texts)
    return await client.system_one(state=state, questions=questions)


async def adecide_batch(class_names, logits, gate_weights, texts_per_sample, client=None):
    """
    Async batched version — fires all B per-sample Jev calls concurrently
    instead of sequentially. Each sample's decision is independent, so wall
    time is roughly one call's latency rather than B of them.

    class_names: list mapping class index -> label. logits: (B, num_classes).
    gate_weights: (B, 2). texts_per_sample: list of length B, each a list of
    strings (from detect_and_read_text).

    Returns a list of length B of Jev responses, one per sample, in order.
    """
    import asyncio
    import torch

    if client is None:
        client = _get_async_client()

    probs = torch.softmax(logits, dim=1)
    top_prob, top_idx = probs.max(dim=1)

    calls = []
    for i in range(logits.shape[0]):
        class_name = class_names[top_idx[i].item()]
        class_prob = top_prob[i].item()
        gw = (gate_weights[i, 0].item(), gate_weights[i, 1].item())
        calls.append(adecide(class_name, class_prob, gw, texts_per_sample[i], client=client))
    return await asyncio.gather(*calls)


def decide_batch(class_names, logits, gate_weights, texts_per_sample, client=None):
    """
    Sync convenience wrapper around adecide_batch — same signature and
    return value as before, but the underlying calls now run concurrently.
    `client`, if given, must be an AsyncTypeSafeClient (a plain sync
    TypeSafeClient won't work here; pass None to let it build one).

    Only usable outside a running event loop (plain scripts, notebooks
    without an active loop). If you're already in async code, call
    `await adecide_batch(...)` directly instead.
    """
    import asyncio

    return asyncio.run(
        adecide_batch(class_names, logits, gate_weights, texts_per_sample, client=client)
    )


def detect_read_and_decide(model, gray_lowres, color_highres, class_names,
                            engine="tesseract", lang="eng", reader=None,
                            mask_threshold=0.5, min_area=20, client=None):
    """
    Full pipeline: classify + localize + OCR (detect_and_read_text), then
    turn the result into one calibrated Jev decision per sample.

    Returns (logits, gate_weights, texts_per_sample, jev_decisions).
    """
    from dual_pathway_classifier import detect_and_read_text

    logits, gate_weights, texts_per_sample = detect_and_read_text(
        model, gray_lowres, color_highres, engine=engine, lang=lang,
        reader=reader, mask_threshold=mask_threshold, min_area=min_area,
    )
    jev_decisions = decide_batch(class_names, logits, gate_weights, texts_per_sample, client=client)
    return logits, gate_weights, texts_per_sample, jev_decisions
