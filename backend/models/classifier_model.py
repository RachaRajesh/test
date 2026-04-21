"""
EfficientNet-B3 loader (optional).

If you train/download EfficientNet-B3 weights for 10-class photo quality,
place them at backend/models/weights/efficientnet_b3_quality.pth and call
load_classifier_model() from services/classifier.py.

Currently the services/classifier.py uses classical CV heuristics and does
NOT call this. This stub is here so the wiring is ready.
"""
import os

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "efficientnet_b3_quality.pth")


def load_classifier_model(weights_path: str | None = None):
    import torch
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=(weights_path is None),
        num_classes=10,
    )
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device).eval()
    return model, device
