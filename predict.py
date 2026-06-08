"""
predict.py — MobileNetV3 binary health classifier (fully offline, no external API).
Delegates disease details to disease_classifier.py after binary classification.
"""

import os, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from disease_classifier import predict as classify_disease

MODEL_PATH   = "model/dog_health_model.pth"
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model       = None
_class_names = ["healthy", "infected"]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _load_model():
    global _model, _class_names
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        _class_names = ckpt.get("classes", _class_names)
        print(f"[predict] Loaded model — classes: {_class_names}")
    else:
        print("[predict] WARNING: No model found — demo mode.")
    net.eval().to(DEVICE)
    _model = net


def predict(image_path: str) -> dict:
    """
    Run binary classification + offline disease analysis.
    Returns a result dict ready to be sent as JSON to the frontend.
    """
    global _model
    if _model is None:
        _load_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    probs_dict = {cls: round(probs[i].item() * 100, 2)
                  for i, cls in enumerate(_class_names)}
    pred_idx   = probs.argmax().item()
    label      = _class_names[pred_idx]
    confidence = round(probs[pred_idx].item() * 100, 2)

    # Delegate everything (including disease analysis) to disease_classifier
    result = classify_disease(image_path)

    # Ensure binary confidence from this model is always present
    result["confidence"]    = confidence
    result["probabilities"] = probs_dict

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    r = predict(sys.argv[1])
    print(json.dumps(r, indent=2))