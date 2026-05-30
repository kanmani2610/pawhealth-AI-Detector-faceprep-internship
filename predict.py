"""
predict.py — MobileNetV3 binary health classifier + Claude vision for disease details.
"""

import os, base64, json, urllib.request, urllib.error
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "model/dog_health_model.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model       = None
_class_names = ["healthy", "infected"]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Disease knowledge base (fallback if API unavailable) ─────────────────────
DISEASE_KB = {
    "mange": {
        "display": "Mange (Scabies/Demodectic)",
        "symptoms": ["Hair loss", "Scaly/crusty skin", "Intense itching", "Red inflamed patches"],
        "precautions": [
            "Isolate the dog immediately from other animals",
            "Wear gloves when handling — sarcoptic mange is zoonotic",
            "Contact a vet for antiparasitic treatment (ivermectin/selamectin)",
            "Disinfect bedding and surroundings",
            "Do not allow contact with healthy dogs until cleared",
        ],
        "urgency": "high",
    },
    "distemper": {
        "display": "Canine Distemper",
        "symptoms": ["Discharge from eyes/nose", "Lethargy", "Coughing", "Fever"],
        "precautions": [
            "Keep the dog away from vaccinated dogs too — highly contagious",
            "Seek emergency veterinary care immediately",
            "Supportive care: fluids, nutrition",
            "No specific cure — prevention via vaccination",
            "Wear PPE when handling",
        ],
        "urgency": "critical",
    },
    "parvovirus": {
        "display": "Canine Parvovirus",
        "symptoms": ["Severe vomiting", "Bloody diarrhoea", "Lethargy", "Loss of appetite"],
        "precautions": [
            "Emergency vet care required — high fatality if untreated",
            "Strict isolation from other dogs",
            "Bleach-disinfect any surfaces the dog contacts",
            "IV fluids and antibiotics needed",
            "Virus survives months in environment — deep-clean area",
        ],
        "urgency": "critical",
    },
    "skin_infection": {
        "display": "Bacterial / Fungal Skin Infection",
        "symptoms": ["Lesions or sores", "Odour", "Swelling", "Discharge"],
        "precautions": [
            "Contact animal welfare / vet for topical or oral antibiotics",
            "Keep wounds clean and dry",
            "Prevent the dog from licking wounds",
            "Wear gloves during handling",
            "Monitor for spread or worsening",
        ],
        "urgency": "medium",
    },
    "tick_infestation": {
        "display": "Tick / Flea Infestation",
        "symptoms": ["Visible ticks or fleas", "Constant scratching", "Skin irritation", "Anaemia in severe cases"],
        "precautions": [
            "Use vet-approved antiparasitic treatment",
            "Remove ticks carefully with tweezers — do not crush",
            "Treat environment to break flea life cycle",
            "Check for tick-borne diseases (ehrlichiosis, babesiosis)",
            "Wear gloves when handling heavily infested animals",
        ],
        "urgency": "medium",
    },
    "wound_injury": {
        "display": "Open Wound / Injury",
        "symptoms": ["Visible cuts or lacerations", "Limping", "Blood", "Swelling"],
        "precautions": [
            "Apply gentle pressure to control bleeding",
            "Do not attempt to clean deep wounds yourself",
            "Contact a vet or animal rescue immediately",
            "Prevent infection by limiting contact with dirt/water",
            "Watch for signs of shock (pale gums, rapid breathing)",
        ],
        "urgency": "high",
    },
    "eye_infection": {
        "display": "Ocular / Eye Infection",
        "symptoms": ["Discharge from eyes", "Redness", "Swelling around eyes", "Squinting"],
        "precautions": [
            "Wipe discharge gently with a damp cloth",
            "Do not apply human eye drops",
            "Contact a vet for appropriate eye drops",
            "Prevent rubbing by using a cone if available",
            "Monitor for progression to blindness",
        ],
        "urgency": "medium",
    },
    "unknown_infection": {
        "display": "Infection (Type Undetermined)",
        "symptoms": ["Visible signs of illness", "Abnormal appearance", "Possible distress"],
        "precautions": [
            "Contact a local animal welfare organisation immediately",
            "Do not handle without gloves",
            "Provide fresh water and shade if safe to do so",
            "Photograph and report to nearest NGO or vet",
            "Do not attempt home treatment without diagnosis",
        ],
        "urgency": "high",
    },
}

URGENCY_META = {
    "critical": {"label": "CRITICAL — Vet Immediately", "color": "#ff4444"},
    "high":     {"label": "HIGH — Needs Prompt Care",   "color": "#ff8c00"},
    "medium":   {"label": "MEDIUM — Schedule Vet Visit","color": "#f0c040"},
}


# ── Model loading ─────────────────────────────────────────────────────────────
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


# ── Vision analysis via Claude API ───────────────────────────────────────────
def _analyse_disease_with_claude(image_path: str) -> dict:
    """
    Send the image to Claude vision. Ask it to identify which disease type
    from our knowledge base best matches, and return structured JSON.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None   # fall back to KB lookup

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = image_path.rsplit(".", 1)[-1].lower()
    media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png",  "webp": "image/webp", "gif": "image/gif"}
    media_type = media_map.get(ext, "image/jpeg")

    disease_list = list(DISEASE_KB.keys())

    system = (
        "You are a veterinary AI assistant. Analyse dog health images and return ONLY "
        "valid JSON — no markdown, no explanation outside the JSON."
    )
    user_prompt = (
        f"Look at this dog image carefully. The dog has been flagged as INFECTED by a classifier.\n\n"
        f"Choose the single best-matching disease from this list: {disease_list}\n\n"
        f"Also rate overall severity: low | medium | high | critical\n\n"
        f"Return ONLY this JSON structure:\n"
        f'{{"disease_key": "<key from list>", "confidence_note": "<1 sentence why>", '
        f'"visible_symptoms": ["<symptom1>", "<symptom2>", "<symptom3>"], '
        f'"severity": "<low|medium|high|critical>"}}'
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 400,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": user_prompt},
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"].strip()
        # Strip any accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[predict] Claude API error: {e}")
        return None


# ── Main predict function ─────────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    global _model
    if _model is None:
        _load_model()

    # 1. Binary classification
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

    result = {
        "label":         label,
        "confidence":    confidence,
        "probabilities": probs_dict,
        "disease":       None,
    }

    # 2. Disease details — only if infected
    if label == "infected":
        claude_data = _analyse_disease_with_claude(image_path)

        if claude_data:
            key  = claude_data.get("disease_key", "unknown_infection")
            kb   = DISEASE_KB.get(key, DISEASE_KB["unknown_infection"])
            urgency = claude_data.get("severity", kb["urgency"])
            result["disease"] = {
                "key":              key,
                "display_name":     kb["display"],
                "confidence_note":  claude_data.get("confidence_note", ""),
                "visible_symptoms": claude_data.get("visible_symptoms", kb["symptoms"]),
                "precautions":      kb["precautions"],
                "urgency":          urgency,
                "urgency_label":    URGENCY_META.get(urgency, URGENCY_META["high"])["label"],
                "urgency_color":    URGENCY_META.get(urgency, URGENCY_META["high"])["color"],
                "source":           "ai",
            }
        else:
            # Fallback: pick most likely disease from KB
            kb = DISEASE_KB["unknown_infection"]
            result["disease"] = {
                "key":              "unknown_infection",
                "display_name":     kb["display"],
                "confidence_note":  "Visual analysis unavailable — general guidance shown.",
                "visible_symptoms": kb["symptoms"],
                "precautions":      kb["precautions"],
                "urgency":          kb["urgency"],
                "urgency_label":    URGENCY_META[kb["urgency"]]["label"],
                "urgency_color":    URGENCY_META[kb["urgency"]]["color"],
                "source":           "fallback",
            }

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    r = predict(sys.argv[1])
    print(json.dumps(r, indent=2))