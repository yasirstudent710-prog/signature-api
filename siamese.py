# -*- coding: utf-8 -*-
"""
Siamese verification model — API ke liye loader.

Ye module bilkul WOHI architecture dobara banata hai jo training notebook
(Siemes_Network_Model.ipynb) me train hui thi, taake us ki saved weights
(siamese.weights.h5) load ho saken:

  Embedding CNN (shared): Input 128x128x1 grayscale
    Conv2D 32 -> MaxPool -> Conv2D 64 -> MaxPool -> Conv2D 128
    -> GlobalAveragePooling -> Dense 128
  Siamese head: |emb(a) - emb(b)| -> Dense(1, sigmoid)
  Output: 0..1 probability (1 = same person / genuine)

Notebook ke verified results (person-wise split, naye logon par test):
  TEST accuracy 96.7% @ threshold 0.5

Weights file: is folder me "siamese.weights.h5" rakhein (Colab se download
karke). File na ho to verification purani tarah imandari se
VERIFICATION_MODEL_NOT_CONNECTED ke saath failed hoti hai — koi nakli
score nahi banta.

TensorFlow lazily import hota hai (sirf pehli verification par), taake
baqi API (detection/auth) TF ke baghair bhi chalti rahe. TF install karne
ke liye: pip install -r requirements-verification.txt
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NAME = "siamese.weights.h5"
WEIGHTS_PATH = os.path.join(BASE_DIR, WEIGHTS_NAME)

# Verdict bands (API-side policy, notebook ke 0.5 threshold ke gird):
#   score >= 55 -> genuine | score <= 45 -> forged | darmiyan -> inconclusive
THRESHOLD = 50.0
GENUINE_AT = 55.0
FORGED_AT = 45.0

_model = None
_load_error = None


def weights_present() -> bool:
    return os.path.exists(WEIGHTS_PATH)


def _build_model():
    """Notebook wali exact architecture (layer order same hona zaroori hai
    taake weights sahi jagah load hon)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    def embedding_net():
        inp = layers.Input((128, 128, 1))
        x = layers.Conv2D(32, 3, activation="relu")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        return models.Model(inp, x, name="embedding")

    base = embedding_net()
    in_a = layers.Input((128, 128, 1), name="a")
    in_b = layers.Input((128, 128, 1), name="b")
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([base(in_a), base(in_b)])
    out = layers.Dense(1, activation="sigmoid")(diff)
    model = models.Model([in_a, in_b], out)
    model.load_weights(WEIGHTS_PATH)
    return model


def _get_model():
    global _model, _load_error
    if _model is not None:
        return _model
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        _model = _build_model()
    except ImportError:
        _load_error = (
            "TensorFlow installed nahi hai — verification ke liye chalayen: "
            "pip install -r requirements-verification.txt"
        )
        raise RuntimeError(_load_error)
    except Exception as e:
        _load_error = f"Siamese weights load nahi ho sakin ({WEIGHTS_NAME}): {e}"
        raise RuntimeError(_load_error)
    return _model


def _load_image(path: str):
    """Notebook wali preprocessing: grayscale, 128x128, 0..1 float."""
    import numpy as np
    from PIL import Image

    arr = np.array(
        Image.open(path).convert("L").resize((128, 128)), dtype="float32"
    )
    return (arr / 255.0)[None, ..., None]  # (1, 128, 128, 1)


def verify_pair(original_path: str, suspected_path: str) -> dict:
    """Do signature images compare karke contract-wala result dein."""
    model = _get_model()
    a = _load_image(original_path)
    b = _load_image(suspected_path)
    prob = float(model.predict([a, b], verbose=0)[0][0])
    score = round(prob * 100.0, 1)
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
            "model": "Siamese CNN (128x128 grayscale)",
            "test_accuracy_percent": 96.7,
        },
    }
