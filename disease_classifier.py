"""
disease_classifier.py
---------------------
Offline multi-class dog disease classifier.

Stage 1: MobileNetV3 binary model -> healthy / infected
Stage 2: Deep feature extraction + colour/texture analysis
         -> rule-based disease classification.
"""

import os, math
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "model", "dog_health_model.pth")
IMG_SIZE          = 224
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── shared transform ─────────────────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── lazy-loaded models ───────────────────────────────────────────────────────
_binary_model   = None
_feature_model  = None
_class_names    = ["healthy", "infected"]


def _load_binary():
    global _binary_model, _class_names
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
    if os.path.exists(BINARY_MODEL_PATH):
        ckpt = torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        _class_names = ckpt.get("classes", _class_names)
    net.eval().to(DEVICE)
    _binary_model = net
    print(f"[classifier] Binary model loaded — classes: {_class_names}")


def _load_feature_extractor():
    """Strip the classifier head; use the 576-d features for disease logic."""
    global _feature_model
    net = models.mobilenet_v3_small(weights=None)
    # Keep everything up to the adaptive pool; drop classifier
    net.classifier = nn.Identity()
    net.eval().to(DEVICE)
    _feature_model = net
    print("[classifier] Feature extractor loaded (MobileNetV3 backbone)")


# ─── image analysis helpers ───────────────────────────────────────────────────

def _rgb_stats(img: Image.Image) -> dict:
    """Mean and stddev per channel (R,G,B) normalised 0-1."""
    stat = ImageStat.Stat(img)
    mn   = [v / 255.0 for v in stat.mean[:3]]
    sd   = [v / 255.0 for v in stat.stddev[:3]]
    return {"mean_r": mn[0], "mean_g": mn[1], "mean_b": mn[2],
            "std_r":  sd[0], "std_g":  sd[1], "std_b":  sd[2]}


def _texture_score(img: Image.Image) -> float:
    """
    Laplacian variance as a proxy for skin roughness / patchiness.
    High value  → rough / patchy / crusted skin (mange, ringworm)
    Low value   → smooth fur
    """
    gray = np.array(img.convert("L").resize((64, 64)), dtype=np.float32)
    lap  = (
        -gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
        + 4 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _brightness(img: Image.Image) -> float:
    stat = ImageStat.Stat(img.convert("L"))
    return stat.mean[0] / 255.0


def _redness_ratio(img: Image.Image) -> float:
    """Fraction of pixels where R channel dominates significantly (inflamed skin)."""
    arr = np.array(img.resize((64, 64)), dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    red_mask = (r > 120) & (r - g > 30) & (r - b > 30)
    return float(red_mask.sum()) / (64 * 64)


def _dark_spot_ratio(img: Image.Image) -> float:
    """Fraction of dark pixels — proxy for bald patches / dark lesions."""
    gray = np.array(img.convert("L").resize((64, 64)), dtype=np.float32)
    return float((gray < 60).sum()) / (64 * 64)


def _eye_region_brightness(img: Image.Image) -> float:
    """
    Crop upper-middle third of image (approximate eye region).
    High brightness deviation → discharge / swelling.
    """
    w, h = img.size
    region = img.crop((w//4, h//8, 3*w//4, h//3))
    return _brightness(region)


def _extract_features(img: Image.Image) -> dict:
    """Combine deep-feature L2 norm statistics with hand-crafted image stats."""
    global _feature_model
    if _feature_model is None:
        _load_feature_extractor()

    tensor = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = _feature_model(tensor).squeeze()          # (576,) or (1, 576, 1, 1)
        if feat.dim() > 1:
            feat = feat.view(-1)
        feat = feat.cpu().numpy()

    # Partition features into 8 equal buckets → energy per bucket
    k    = len(feat) // 8
    energy = [float(np.sum(feat[i*k:(i+1)*k]**2)) for i in range(8)]
    total  = max(sum(energy), 1e-6)
    energy = [e / total for e in energy]                 # normalise

    return {
        "feat_energy": energy,
        **_rgb_stats(img),
        "texture":     _texture_score(img),
        "redness":     _redness_ratio(img),
        "dark_spots":  _dark_spot_ratio(img),
        "brightness":  _brightness(img),
        "eye_bright":  _eye_region_brightness(img),
    }


# ─── rule-based disease scorer ────────────────────────────────────────────────

def _score_diseases(feats: dict) -> dict:
    """
    Returns a score dict {disease: score} using hand-crafted rules
    derived from visual characteristics of each condition.
    Scores are positive floats; higher = more likely.
    """
    tx  = feats["texture"]
    rd  = feats["redness"]
    ds  = feats["dark_spots"]
    br  = feats["brightness"]
    eb  = feats["eye_bright"]
    mr  = feats["mean_r"]
    mg  = feats["mean_g"]
    mb  = feats["mean_b"]
    sr  = feats["std_r"]
    en  = feats["feat_energy"]   # list of 8 normalised energies

    scores = {}

    # ── Mange (Sarcoptic / Demodectic) ────────────────────────────────────────
    # High texture variance (rough/crusted), dark spots (bald patches), redness
    scores["mange"] = (
        0.35 * min(tx / 400.0, 1.0)       # rough skin
      + 0.30 * min(ds / 0.25, 1.0)        # bald/dark patches
      + 0.20 * min(rd / 0.15, 1.0)        # inflamed areas
      + 0.15 * min(sr / 0.18, 1.0)        # colour variation
    )

    # ── Ringworm (Dermatophytosis) ────────────────────────────────────────────
    # Circular patches, moderate texture, pale/grey discolouration
    scores["ringworm"] = (
        0.30 * min(tx / 350.0, 1.0)
      + 0.25 * (1.0 - min(mr / 0.55, 1.0))  # less red → grey/pale
      + 0.25 * min(ds / 0.20, 1.0)
      + 0.20 * min(en[2] / 0.18, 1.0)       # mid-freq feature energy
    )

    # ── Distemper ─────────────────────────────────────────────────────────────
    # Nasal/eye discharge, dull coat, overall low brightness, high eye-region brightness
    scores["distemper"] = (
        0.35 * min(eb / 0.75, 1.0)         # bright eye region (discharge)
      + 0.25 * (1.0 - min(br / 0.55, 1.0)) # overall dull/dark
      + 0.20 * min(en[0] / 0.20, 1.0)      # low-freq feature energy
      + 0.20 * min(rd / 0.10, 1.0)         # mild redness
    )

    # ── Parvovirus ────────────────────────────────────────────────────────────
    # Very lethargic / curled posture; overall low brightness; little colour variation
    scores["parvovirus"] = (
        0.40 * (1.0 - min(br / 0.40, 1.0)) # very dark/dull image
      + 0.30 * (1.0 - min(sr / 0.12, 1.0)) # low colour variation (grey)
      + 0.30 * min(en[1] / 0.18, 1.0)
    )

    # ── Tick / Flea Infestation ───────────────────────────────────────────────
    # Small dark dots on coat, moderate texture, normal fur colour
    scores["tick_infestation"] = (
        0.40 * min(ds / 0.15, 1.0)         # dark tick bodies
      + 0.30 * min(tx / 300.0, 1.0)        # slight roughness
      + 0.30 * (1.0 - min(rd / 0.10, 1.0)) # NOT heavily red (not wound)
    )

    # ── Wound / Injury ────────────────────────────────────────────────────────
    # High redness, bright spot in frame, possibly high contrast
    scores["wound_injury"] = (
        0.45 * min(rd / 0.20, 1.0)         # strong redness
      + 0.30 * min(tx / 300.0, 1.0)        # rough/irregular region
      + 0.25 * min(mr / 0.50, 1.0)         # overall warm/red image
    )

    # ── Eye Infection ─────────────────────────────────────────────────────────
    # Very bright upper region (swollen/watery eyes), normal body
    scores["eye_infection"] = (
        0.50 * min(eb / 0.80, 1.0)
      + 0.30 * min(rd / 0.08, 1.0)
      + 0.20 * (1.0 - abs(br - 0.55))      # medium overall brightness
    )

    # ── Bacterial Skin Infection ──────────────────────────────────────────────
    # Patchy lesions, moderate redness, odour not visible but texture indicates
    scores["skin_infection"] = (
        0.30 * min(rd / 0.12, 1.0)
      + 0.35 * min(tx / 320.0, 1.0)
      + 0.20 * min(ds / 0.18, 1.0)
      + 0.15 * min(en[4] / 0.15, 1.0)
    )

    return scores


# ─── disease knowledge base ───────────────────────────────────────────────────

DISEASE_KB = {
    "mange": {
        "display":    "Mange (Sarcoptic / Demodectic)",
        "urgency":    "high",
        "symptoms":   [
            "Severe hair loss in patches",
            "Thick, crusty, or scaly skin",
            "Intense itching and self-scratching",
            "Red, inflamed skin around ears, elbows, belly",
            "Skin may appear grey or yellowish"
        ],
        "basic_cure": [
            "Oral isoxazoline (fluralaner/afoxolaner/sarolaner) — vet prescription, given monthly; now first-line treatment",
            "Topical spot-on: imidacloprid 10% + moxidectin 2.5% (Advocate) OR selamectin 12% — vet prescription",
            "Lime sulfur dip twice weekly for accessible lesions — dilute 1:16 with water",
            "Medicated shampoo: miconazole 2% + chlorhexidine 2% — bath twice weekly",
            "Confirm cure with vet skin scraping after minimum 4 weeks of treatment",
            "Provide high-protein diet to support skin regeneration"
        ],
        "home_first_aid": [
            "Wear gloves — sarcoptic mange CAN transfer to humans",
            "Isolate from other dogs immediately",
            "Apply diluted neem + coconut oil to affected areas",
            "Provide fresh water and shade"
        ],
        "is_contagious": True,
        "zoonotic": True
    },
    "ringworm": {
        "display":    "Ringworm (Dermatophytosis)",
        "urgency":    "medium",
        "symptoms":   [
            "Circular bald patches with defined edges",
            "Scaly, grey or crusty skin within patches",
            "Patches mainly on face, ears, paws",
            "Mild itching (less than mange)",
            "Broken hair stubs at edges of patches"
        ],
        "basic_cure": [
            "Apply antifungal cream (miconazole 2% / clotrimazole) twice daily to lesions",
            "Oral antifungal (itraconazole preferred over griseofulvin; terbinafine also used) — vet prescription",
            "Antifungal shampoo (miconazole + chlorhexidine) twice weekly — leave on 5–10 min before rinsing",
            "Clip fur around lesions to improve airflow and topical penetration",
            "Continue treatment minimum 6 weeks; severe cases may require longer — confirm cure with vet"
        ],
        "home_first_aid": [
            "Wash hands thoroughly — ringworm is contagious to humans",
            "Do not let children touch affected areas",
            "Apply diluted apple cider vinegar or coconut oil as short-term relief",
            "Keep the dog's environment clean and dry"
        ],
        "is_contagious": True,
        "zoonotic": True
    },
    "distemper": {
        "display":    "Canine Distemper",
        "urgency":    "critical",
        "symptoms":   [
            "Discharge from eyes and nose",
            "Coughing, laboured breathing",
            "High fever and lethargy",
            "Loss of appetite and vomiting",
            "Neurological twitching in advanced stage"
        ],
        "basic_cure": [
            "NO specific cure — supportive treatment only",
            "IV fluids to prevent dehydration (vet hospital)",
            "Antibiotics to prevent secondary bacterial infections",
            "Anti-seizure medication for neurological symptoms",
            "Nutritional support and isolation"
        ],
        "home_first_aid": [
            "Rush to vet IMMEDIATELY — high fatality without treatment",
            "Keep dog warm, dry and isolated",
            "Offer small amounts of water frequently",
            "Do NOT attempt home treatment — requires hospitalisation"
        ],
        "is_contagious": True,
        "zoonotic": False
    },
    "parvovirus": {
        "display":    "Canine Parvovirus (CPV)",
        "urgency":    "critical",
        "symptoms":   [
            "Severe vomiting (may contain blood)",
            "Bloody, foul-smelling diarrhoea",
            "Extreme lethargy — unable to stand",
            "Rapid weight loss",
            "Pale or yellow gums"
        ],
        "basic_cure": [
            "EMERGENCY VET CARE REQUIRED — >90% fatal without IV fluids; survival rate 68–90% with prompt hospitalised treatment",
            "IV fluid therapy to correct dehydration and electrolyte loss",
            "IV antibiotics (ampicillin/enrofloxacin)",
            "Anti-nausea medication (ondansetron/maropitant)",
            "Plasma transfusion in severe cases"
        ],
        "home_first_aid": [
            "Go to emergency vet within the HOUR",
            "Do NOT feed — NPO (nothing by mouth)",
            "Keep the dog warm and still",
            "Bleach-disinfect any area the dog touched (virus survives months)"
        ],
        "is_contagious": True,
        "zoonotic": False
    },
    "tick_infestation": {
        "display":    "Tick / Flea Infestation",
        "urgency":    "medium",
        "symptoms":   [
            "Visible ticks embedded in skin (ears, neck, paws)",
            "Constant scratching and restlessness",
            "Small red bite marks on skin",
            "Anaemia in heavy infestations (pale gums)",
            "Flea dirt (dark specks) visible on coat"
        ],
        "basic_cure": [
            "Apply topical antiparasitic (fipronil spot-on or permethrin spray)",
            "Remove ticks manually with tweezers — grasp close to skin, twist gently",
            "Oral antiparasitics (afoxolaner / fluralaner) — vet prescription",
            "Anti-flea shampoo bath followed by flea comb",
            "Treat environment (bedding, surroundings) with permethrin spray"
        ],
        "home_first_aid": [
            "Never squeeze or crush ticks — use tweezers",
            "Apply petroleum jelly or coconut oil on ticks before removal",
            "Check for fever or joint swelling — signs of tick-borne disease",
            "Burn or drown removed ticks — do not crush with fingers"
        ],
        "is_contagious": False,
        "zoonotic": True
    },
    "wound_injury": {
        "display":    "Open Wound / Injury",
        "urgency":    "high",
        "symptoms":   [
            "Visible cuts, lacerations, or puncture wounds",
            "Bleeding (may be dried/clotted)",
            "Limping or reluctance to bear weight",
            "Swelling around wound site",
            "Signs of pain when touched"
        ],
        "basic_cure": [
            "Irrigate wound immediately with sterile saline or clean tap water — use a 20 mL syringe for pressure flushing",
            "Do NOT use betadine/hydrogen peroxide on open wounds — delays healing; saline only",
            "Bandage loosely with non-stick dressing — change daily",
            "Antibiotics if wound infected: amoxicillin-clavulanate (Clavamox) or cephalexin — vet prescription",
            "Tetanus prophylaxis may be required for deep puncture wounds",
            "Monitor daily for signs of infection: heat, pus, swelling, odour"
        ],
        "home_first_aid": [
            "Apply gentle pressure with clean cloth to stop bleeding",
            "Rinse with clean running water only — do NOT apply betadine, hydrogen peroxide, or human antiseptic creams",
            "Prevent the dog from licking the wound — improvise a collar if needed",
            "Keep the dog calm and still until vet help arrives"
        ],
        "is_contagious": False,
        "zoonotic": False
    },
    "eye_infection": {
        "display":    "Ocular Infection / Conjunctivitis",
        "urgency":    "medium",
        "symptoms":   [
            "Yellow or green discharge from one or both eyes",
            "Redness and swelling around the eye",
            "Squinting or keeping eye partially closed",
            "Crusting on eyelids",
            "Pawing at eyes"
        ],
        "basic_cure": [
            "Rinse eyes 1–2× daily with sterile eye irrigation solution (saline) — most cases resolve with this alone",
            "Apply lubricating eye gel (hypromellose) before walks as a protective barrier",
            "E-collar (cone) to prevent the dog rubbing its eyes",
            "Vet may prescribe topical antihistamine (ketotifen) if allergic origin is suspected",
            "Antibiotic eye drops (chloramphenicol/tobramycin) ONLY if bacterial infection is confirmed by vet — not routine first-line",
            "Rule out underlying cause: distemper, entropion, foreign body, dry eye (KCS)"
        ],
        "home_first_aid": [
            "Use clean cotton dipped in cooled boiled water to wipe discharge",
            "Do NOT use human eye drops",
            "Keep the dog's face clean and dry",
            "See vet if no improvement in 48 hours"
        ],
        "is_contagious": False,
        "zoonotic": False
    },
    "skin_infection": {
        "display":    "Bacterial Skin Infection (Pyoderma)",
        "urgency":    "medium",
        "symptoms":   [
            "Red, bumpy, or pus-filled lesions",
            "Foul odour from skin",
            "Crusting and scaling",
            "Hair loss in affected areas",
            "Dog constantly biting or licking skin"
        ],
        "basic_cure": [
            "Antibacterial shampoo bath (chlorhexidine 2–4%) twice weekly",
            "Oral antibiotics for 3–6 weeks (amoxicillin-clavulanate — vet only)",
            "Topical antibiotic spray (mupirocin)",
            "Treat underlying cause (allergy, hormonal — requires vet diagnosis)",
            "Medicated wipes for maintenance between baths"
        ],
        "home_first_aid": [
            "Keep affected areas clean and dry",
            "Diluted apple cider vinegar spray as temporary antifungal/antibacterial",
            "Apply coconut oil to soothe irritated skin",
            "Prevent the dog from licking — use an e-collar if available"
        ],
        "is_contagious": False,
        "zoonotic": False
    }
}

URGENCY_META = {
    "critical": {"label": "CRITICAL — Emergency Vet Now", "color": "#dc2626"},
    "high":     {"label": "HIGH — Vet Within 24 Hours",   "color": "#ea580c"},
    "medium":   {"label": "MEDIUM — Vet This Week",       "color": "#d97706"},
    "low":      {"label": "LOW — Monitor & Observe",      "color": "#65a30d"},
}


# ─── main predict function ────────────────────────────────────────────────────

def predict(image_path: str) -> dict:
    global _binary_model
    if _binary_model is None:
        _load_binary()

    img    = Image.open(image_path).convert("RGB")
    tensor = _tf(img).unsqueeze(0).to(DEVICE)

    # ── Stage 1: binary classification ────────────────────────────────────────
    with torch.no_grad():
        logits = _binary_model(tensor)
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

    if label != "infected":
        return result

    # ── Stage 2: disease classification (offline) ──────────────────────────────
    feats   = _extract_features(img)
    scores  = _score_diseases(feats)

    # Softmax over scores for interpretable percentages
    vals    = list(scores.values())
    keys    = list(scores.keys())
    exp_v   = [math.exp(v * 8) for v in vals]          # temperature 1/8
    total   = sum(exp_v)
    soft    = [e / total for e in exp_v]

    disease_probs = {k: round(s * 100, 1) for k, s in zip(keys, soft)}
    top_key       = max(scores, key=scores.get)
    top_kb        = DISEASE_KB[top_key]
    urgency       = top_kb["urgency"]

    # Top-3 for display
    top3 = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_display = [
        {"key": k, "name": DISEASE_KB[k]["display"], "pct": p}
        for k, p in top3
    ]

    result["disease"] = {
        "key":            top_key,
        "display_name":   top_kb["display"],
        "urgency":        urgency,
        "urgency_label":  URGENCY_META[urgency]["label"],
        "urgency_color":  URGENCY_META[urgency]["color"],
        "symptoms":       top_kb["symptoms"],
        "basic_cure":     top_kb["basic_cure"],
        "home_first_aid": top_kb["home_first_aid"],
        "is_contagious":  top_kb["is_contagious"],
        "zoonotic":       top_kb["zoonotic"],
        "top3_diseases":  top3_display,
        "disease_probs":  disease_probs,
    }
    return result


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python disease_classifier.py <image>")
        sys.exit(1)
    r = predict(sys.argv[1])
    print(json.dumps(r, indent=2))
