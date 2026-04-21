"""
services/classifier.py — Photo quality classifier.

Uses classical CV (Laplacian variance, brightness/contrast stats, Haar face
detection) with carefully tuned weights. This is NOT EfficientNet — the
frontend badge has been updated accordingly. To swap in a trained EfficientNet-B3,
implement load_classifier_model() in models/classifier_model.py and call it here.

Scoring (1-10):
    - Exposure (brightness away from 0.5)   : 2.0 pts
    - Contrast (std dev of luminance)       : 2.0 pts
    - Sharpness (Laplacian variance)        : 2.5 pts
    - Noise (low-noise bonus)               : 1.0 pts
    - Face-detection bonus                  : 0.5 pts
    - Base                                  : 2.0 pts
Max: 10.0
"""
import numpy as np
import cv2
from PIL import Image


# Tunable thresholds. Adjusted from the original values — original had
# sharpness/400 which only maxes out on very sharp photos; this gives a
# gentler curve so typical phone photos don't all score 5-6.
SHARPNESS_SOFT_MAX = 250.0   # Laplacian var where sharpness credit caps
CONTRAST_SOFT_MAX  = 0.40    # luminance std (0-1) where contrast credit caps
NOISE_PENALTY_FROM = 4.0     # noise value where penalty starts
NOISE_PENALTY_TO   = 14.0    # noise value where no credit

_face_cascade = None
def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _linear_credit(value: float, soft_max: float) -> float:
    """0..1 linear ramp, clipped."""
    return float(max(0.0, min(1.0, value / soft_max)))


def classify_image(pil_image: Image.Image) -> dict:
    img_np = np.array(pil_image.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- Metrics ---
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean()) / 255.0              # 0..1
    contrast   = float(gray.std()) / 128.0               # ~0..1 typical
    blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
    noise      = float(np.std(gray.astype(float) - blurred.astype(float)))

    faces    = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    has_face = len(faces) > 0

    # --- Score (weighted components) ---
    score = 2.0  # base
    # Exposure: peak at 0.5, falls off on either side.
    # abs(b - 0.5) in [0, 0.5]. 1 - 2*abs gives 1 at 0.5, 0 at 0 or 1.
    exposure_credit = 1.0 - min(1.0, abs(brightness - 0.5) * 2.0)
    score += exposure_credit * 2.0

    score += _linear_credit(contrast, CONTRAST_SOFT_MAX) * 2.0
    score += _linear_credit(sharpness, SHARPNESS_SOFT_MAX) * 2.5

    # Noise: full credit below NOISE_PENALTY_FROM, zero above NOISE_PENALTY_TO
    if noise <= NOISE_PENALTY_FROM:
        noise_credit = 1.0
    elif noise >= NOISE_PENALTY_TO:
        noise_credit = 0.0
    else:
        noise_credit = 1.0 - (noise - NOISE_PENALTY_FROM) / (NOISE_PENALTY_TO - NOISE_PENALTY_FROM)
    score += noise_credit * 1.0

    if has_face:
        score += 0.5

    score_int = int(round(max(1, min(10, score))))

    labels = {
        1: "Very Poor", 2: "Poor",     3: "Below Average", 4: "Below Average",
        5: "Average",   6: "Decent",   7: "Good",          8: "Very Good",
        9: "Excellent", 10: "Professional",
    }
    label = labels[score_int] + " Quality"

    # --- Tags (human-readable feedback) ---
    tags = []
    if brightness > 0.72:    tags.append({"tag": "Overexposed",     "type": "bad"})
    elif brightness < 0.22:  tags.append({"tag": "Underexposed",    "type": "bad"})
    else:                    tags.append({"tag": "Good Exposure",   "type": "good"})

    if contrast >= 0.30:     tags.append({"tag": "Good Contrast",   "type": "good"})
    elif contrast < 0.15:    tags.append({"tag": "Very Low Contrast","type": "bad"})
    else:                    tags.append({"tag": "Flat Contrast",   "type": "bad"})

    if sharpness > 200:      tags.append({"tag": "Sharp Focus",     "type": "good"})
    elif sharpness < 60:     tags.append({"tag": "Blurry",          "type": "bad"})
    elif sharpness < 120:    tags.append({"tag": "Slightly Soft",   "type": "bad"})

    if noise < 4:            tags.append({"tag": "Low Noise",       "type": "good"})
    elif noise > 12:         tags.append({"tag": "High Noise",      "type": "bad"})

    if has_face:             tags.append({"tag": f"{len(faces)} Face(s) Detected", "type": "good"})

    return {
        "score": score_int,
        "score_raw": round(score, 2),
        "label": label,
        "tags": tags,
        "metrics": {
            "sharpness":  round(sharpness, 2),
            "brightness": round(brightness, 3),
            "contrast":   round(contrast, 3),
            "noise":      round(noise, 2),
            "has_face":   has_face,
            "face_count": int(len(faces)),
        },
    }
