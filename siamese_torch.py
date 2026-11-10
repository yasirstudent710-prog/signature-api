# -*- coding: utf-8 -*-
"""
FALLBACK ONLY — do NOT wire into main.py yet. Draft prepared 2026-11-10.

Purpose: if TensorFlow cannot install on Yasir's Python 3.14
(`pip install -r requirements-verification.txt` fails with
"no matching distribution" — stable TF 2.21 ships wheels only for
Python 3.10–3.13; on 3.14 pip falls through to pre-release 2.22.0rc0
+ keras-nightly), verification can still connect WITHOUT TensorFlow,
using the teammate's PyTorch ResNet50 Siamese checkpoint instead.
Torch is already on his PC (YOLO), so nothing big needs installing.

What this loader is built from (all read 2026-11-10, local copies):
- verification.zip (teammate Group 8 component) -> model.py (Verifier),
  verify.py (load_model / compare / prepare_tensor) and
  preprocess_sohi.py (normalize_image), plus one_reference_report.json
  (best epoch 17, validation threshold 0.549, test AUC 0.857, ~22% error).

What is STILL unverified — do not present as working until checked:
1. The Drive file best.pt (272.4 MB, 9 Nov, signature_dt folder) has NOT
   been downloaded or inspected here. Only verification.zip (104 KB) was
   downloaded. The epoch-17 check (checkpoint["epoch"] == 17, matching
   the report) must pass before this file is trusted — a wrong best.pt
   (e.g. a detection checkpoint) would load_state_dict-fail or score
   nonsense.
2. This loader has NOT been run against the real checkpoint — only
   syntax-checked and smoke-tested with random weights (see bottom).
3. Its numbers are the teammate model's (AUC 0.857, ~22% error), NOT
   Yasir's 96.7% Siamese. Never mix the two in the presentation.

How to use (only if the TF route fails):
1. Confirm best.pt epoch 17 (load it, read checkpoint["epoch"]).
2. Copy best.pt into signature-api as `resnet50_best.pt`
   (keep the name distinct from signature_model.pt — never overwrite it).
3. In main.py, swap the verification import to this module and restart.
   The API contract is unchanged: verify_pair() returns the same dict
   shape as siamese.py (similarity_score 0-100, verdict, threshold).
   Health check would need a matching `verification_weights_present`
   pointing at resnet50_best.pt.

Score mapping (teammate model outputs cosine similarity, not a
probability): score = (cosine + 1) / 2 * 100, so the report's 1-reference
threshold 0.549 becomes 77.5. Verdict bands are centred on that
threshold: >= 77.5 genuine, <= 67.5 forged, between = inconclusive.
The 10-point grey zone is this draft's choice, not the teammate's —
the teammate used a single hard threshold.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NAME = "resnet50_best.pt"
WEIGHTS_PATH = os.path.join(BASE_DIR, WEIGHTS_NAME)

# Teammate report: best epoch 17, 1-reference threshold (validation).
EXPECTED_EPOCH = 17
THRESHOLD_COSINE = 0.549
THRESHOLD = round((THRESHOLD_COSINE + 1.0) / 2.0 * 100.0, 1)  # 77.5
GENUINE_AT = THRESHOLD
FORGED_AT = THRESHOLD - 10.0  # 67.5 — grey zone is a draft choice, see header

_model = None
_load_error = None


def weights_present() -> bool:
    return os.path.exists(WEIGHTS_PATH)


class Verifier(nn.Module):
    """Byte-for-byte the teammate's model.py (Group 8 verification.zip)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.project = nn.Linear(2048, 128)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.project(self.backbone(images)), dim=1)

    def forward(self, reference: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        cosine = (self.embed(reference) * self.embed(query)).sum(dim=1)
        return self.scale * cosine + self.bias


def _normalize_image(source: Image.Image) -> Image.Image:
    """Byte-for-byte the teammate's preprocess_sohi.normalize_image."""
    gray = ImageOps.exif_transpose(source).convert("L")
    ink = np.asarray(gray) < 235
    if not ink.any():
        raise ValueError("No visible ink in signature image")
    ys, xs = np.where(ink)
    pad_x = max(8, round((xs.max() - xs.min() + 1) * 0.06))
    pad_y = max(8, round((ys.max() - ys.min() + 1) * 0.10))
    box = (max(0, int(xs.min()) - pad_x), max(0, int(ys.min()) - pad_y),
           min(gray.width, int(xs.max()) + pad_x + 1),
           min(gray.height, int(ys.max()) + pad_y + 1))
    cropped = gray.crop(box)
    cropped.thumbnail((240, 112), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (256, 128), 255)
    canvas.paste(cropped, ((256 - cropped.width) // 2, (128 - cropped.height) // 2))
    return canvas


def _prepare_tensor(image: Image.Image) -> torch.Tensor:
    """Byte-for-byte the teammate's verify.prepare_tensor."""
    normalized = _normalize_image(image)
    array = np.asarray(normalized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _get_model():
    global _model, _load_error
    if _model is not None:
        return _model
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(WEIGHTS_PATH)
        torch.set_num_threads(2)
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        if epoch != EXPECTED_EPOCH:
            raise ValueError(
                f"best.pt epoch is {epoch}, expected {EXPECTED_EPOCH} — "
                "this is not the teammate's verified checkpoint"
            )
        _model = Verifier(pretrained=False)
        _model.load_state_dict(checkpoint["model"])
        _model.eval()
    except Exception as e:
        _load_error = f"ResNet50 verification weights load nahi ho sakin ({WEIGHTS_NAME}): {e}"
        raise RuntimeError(_load_error)
    return _model


@torch.inference_mode()
def verify_pair(original_path: str, suspected_path: str) -> dict:
    """Same return shape as siamese.verify_pair (API contract unchanged)."""
    model = _get_model()
    with Image.open(original_path) as a_img, Image.open(suspected_path) as b_img:
        batch = torch.stack([_prepare_tensor(a_img), _prepare_tensor(b_img)])
    embeddings = model.embed(batch).cpu().numpy()
    cosine = float(np.dot(embeddings[0], embeddings[1]))
    score = round((cosine + 1.0) / 2.0 * 100.0, 1)
    if score >= GENUINE_AT:
        verdict = "genuine"
    elif score <= FORGED_AT:
        verdict = "forged"
    else:
        verdict = "inconclusive"
    return {
        "similarity_score": score,
        "verdict": verdict,
        "threshold": THRESHOLD,
        "metrics": {
            "genuine_probability": score,
            "model": "ResNet50 Siamese (teammate Group 8, cosine similarity)",
            "test_accuracy_percent": None,
            "test_auc": 0.857,
            "note": "Teammate model — test AUC 0.857, ~22% error; "
                    "NOT Yasir's 96.7% Siamese. Do not mix the numbers.",
        },
    }
