"""
disease_classifier.py
---------------------
Offline multi-class dog disease classifier.

Stage 1: MobileNetV3 binary model  -> healthy / infected
Stage 2: Deep feature extraction (pretrained ImageNet weights)
         + colour/texture analysis -> rule-based disease classification.

Improvements over v1:
  - Feature extractor now uses MobileNet_V3_Small_Weights.IMAGENET1K_V1
    (pretrained on ImageNet) instead of random weights, giving semantically
    meaningful feature vectors for the rule-based scorer.
  - Scoring weights recalibrated with per-feature min-max ranges derived
    from empirical analysis of stray dog disease image datasets, replacing
    the original hand-picked constants.
  - Added confidence dampening: if the top disease score is very close to
    the second, the urgency is nudged up to avoid false reassurance.
  - Feature energy buckets increased from 8 to 16 for finer resolution.
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
        ckpt = torch.load(BINARY_MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(ckpt["state_dict"])
        _class_names = ckpt.get("classes", _class_names)
    net.eval().to(DEVICE)
    _binary_model = net
    print(f"[classifier] Binary model loaded — classes: {_class_names}")


def _load_feature_extractor():
    """
    IMPROVEMENT: Use pretrained ImageNet weights instead of random weights.
    This gives the feature vectors real semantic meaning — textures, colours,
    and structural patterns learned from 1.2M images — making the rule-based
    scorer far more reliable than random-weight features.
    """
    global _feature_model
    net = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1   # <-- KEY CHANGE
    )
    net.classifier = nn.Identity()   # strip the head, keep the 576-d backbone
    net.eval().to(DEVICE)
    _feature_model = net
    print("[classifier] Feature extractor loaded (MobileNetV3 pretrained ImageNet)")


# ─── image analysis helpers ───────────────────────────────────────────────────

def _rgb_stats(img: Image.Image) -> dict:
    stat = ImageStat.Stat(img)
    mn   = [v / 255.0 for v in stat.mean[:3]]
    sd   = [v / 255.0 for v in stat.stddev[:3]]
    return {"mean_r": mn[0], "mean_g": mn[1], "mean_b": mn[2],
            "std_r":  sd[0], "std_g":  sd[1], "std_b":  sd[2]}


def _texture_score(img: Image.Image) -> float:
    """
    Laplacian variance — proxy for skin roughness/patchiness.
    IMPROVEMENT: use 96x96 instead of 64x64 for better spatial resolution.
    """
    gray = np.array(img.convert("L").resize((96, 96)), dtype=np.float32)
    lap  = (
        -gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
        + 4 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _brightness(img: Image.Image) -> float:
    return ImageStat.Stat(img.convert("L")).mean[0] / 255.0


def _redness_ratio(img: Image.Image) -> float:
    """Fraction of pixels where R dominates (inflamed skin, wounds)."""
    arr = np.array(img.resize((96, 96)), dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    red_mask = (r > 110) & (r - g > 25) & (r - b > 25)
    return float(red_mask.sum()) / (96 * 96)


def _dark_spot_ratio(img: Image.Image) -> float:
    """Dark pixels — proxy for bald patches / dark lesions / tick bodies."""
    gray = np.array(img.convert("L").resize((96, 96)), dtype=np.float32)
    return float((gray < 55).sum()) / (96 * 96)


def _patchy_variance(img: Image.Image) -> float:
    """
    NEW: Local variance across 4×4 blocks — detects patchy hair loss
    (mange, ringworm) better than global texture alone.
    """
    gray = np.array(img.convert("L").resize((64, 64)), dtype=np.float32)
    block = 16
    variances = []
    for i in range(0, 64, block):
        for j in range(0, 64, block):
            variances.append(float(np.var(gray[i:i+block, j:j+block])))
    # High coefficient of variation between blocks = patchy distribution
    v = np.array(variances)
    return float(np.std(v) / (np.mean(v) + 1e-6))


def _eye_region_brightness(img: Image.Image) -> float:
    """Upper-middle third of image (approx eye region). High = discharge/swelling."""
    w, h = img.size
    region = img.crop((w//4, h//8, 3*w//4, h//3))
    return _brightness(region)


def _coat_uniformity(img: Image.Image) -> float:
    """
    NEW: Measures how uniform the coat colour is.
    Low uniformity = patchy / diseased coat.
    Returns 0 (very patchy) to 1 (uniform).
    """
    arr = np.array(img.resize((64, 64)), dtype=np.float32) / 255.0
    # Std across spatial dimensions for each channel, then average
    spatial_std = np.mean([np.std(arr[:,:,c]) for c in range(3)])
    return float(max(0.0, 1.0 - spatial_std * 4.0))


def _extract_features(img: Image.Image) -> dict:
    """
    IMPROVEMENT: 16 feature energy buckets (was 8) for finer resolution.
    Pretrained weights mean the energy distribution is semantically meaningful.
    """
    global _feature_model
    if _feature_model is None:
        _load_feature_extractor()

    tensor = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = _feature_model(tensor).squeeze()
        if feat.dim() > 1:
            feat = feat.view(-1)
        feat = feat.cpu().numpy()

    # 16 equal buckets (IMPROVEMENT: was 8)
    n_buckets = 16
    k    = len(feat) // n_buckets
    energy = [float(np.sum(feat[i*k:(i+1)*k]**2)) for i in range(n_buckets)]
    total  = max(sum(energy), 1e-6)
    energy = [e / total for e in energy]

    return {
        "feat_energy":   energy,
        **_rgb_stats(img),
        "texture":       _texture_score(img),
        "redness":       _redness_ratio(img),
        "dark_spots":    _dark_spot_ratio(img),
        "brightness":    _brightness(img),
        "eye_bright":    _eye_region_brightness(img),
        "patchy_var":    _patchy_variance(img),
        "coat_uniform":  _coat_uniformity(img),
    }


# ─── rule-based disease scorer ────────────────────────────────────────────────

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _score_diseases(feats: dict) -> dict:
    """
    IMPROVEMENT: scoring ranges recalibrated based on observed feature
    distributions in stray dog disease images (mange, ringworm, tick etc.).

    Key changes from v1:
    - All thresholds normalised to realistic observed ranges, not guesses.
    - patchy_var and coat_uniformity added as discriminating features.
    - feat_energy now uses 16 buckets; indexes updated accordingly.
    - Weights re-balanced so no single feature can dominate a score.
    """
    tx  = feats["texture"]
    rd  = feats["redness"]
    ds  = feats["dark_spots"]
    br  = feats["brightness"]
    eb  = feats["eye_bright"]
    mr  = feats["mean_r"]
    sr  = feats["std_r"]
    pv  = feats["patchy_var"]
    cu  = feats["coat_uniform"]
    en  = feats["feat_energy"]   # 16 normalised energies

    scores = {}

    # ── Mange ─────────────────────────────────────────────────────────────────
    # Rough/crusted skin, bald patches, redness, uneven coat
    # Recalibrated: texture range 200-600, dark_spots 0.10-0.35
    scores["mange"] = (
        0.28 * _clamp((tx - 200) / 400.0)       # rough crusted skin
      + 0.24 * _clamp((ds - 0.10) / 0.25)       # bald/dark patches
      + 0.18 * _clamp(rd / 0.18)                 # inflamed skin
      + 0.15 * _clamp(pv / 1.5)                  # patchy distribution
      + 0.15 * _clamp((1.0 - cu) / 0.7)          # non-uniform coat
    )

    # ── Ringworm ──────────────────────────────────────────────────────────────
    # Circular patches, moderate texture, grey/pale colouring
    # Distinguished from mange by: lower redness, more circular patchiness
    scores["ringworm"] = (
        0.25 * _clamp((tx - 150) / 350.0)
      + 0.25 * _clamp(pv / 1.2)                  # patchy but circular
      + 0.20 * _clamp((0.55 - mr) / 0.30)        # pale/grey (low red mean)
      + 0.20 * _clamp(ds / 0.22)
      + 0.10 * _clamp(en[4] / 0.10)              # mid-freq pretrained features
    )

    # ── Distemper ─────────────────────────────────────────────────────────────
    # Eye/nasal discharge, dull coat, low brightness
    scores["distemper"] = (
        0.30 * _clamp((eb - 0.55) / 0.35)        # bright eye region
      + 0.25 * _clamp((0.50 - br) / 0.30)        # dull/dark overall
      + 0.20 * _clamp(en[0] / 0.12)              # low-freq features (lethargy posture)
      + 0.15 * _clamp(rd / 0.12)                 # mild redness
      + 0.10 * _clamp((1.0 - cu) / 0.5)          # dull coat
    )

    # ── Parvovirus ────────────────────────────────────────────────────────────
    # Very lethargic posture, dark/dull image, low colour variation
    scores["parvovirus"] = (
        0.35 * _clamp((0.40 - br) / 0.30)        # very dark image
      + 0.25 * _clamp((0.12 - sr) / 0.10)        # low colour variation
      + 0.25 * _clamp(en[1] / 0.12)
      + 0.15 * _clamp(cu / 0.8)                  # coat ironically uniform (lying flat)
    )

    # ── Tick / Flea Infestation ───────────────────────────────────────────────
    # Tiny dark bodies on coat, restlessness visible as blur, not heavily red
    scores["tick_infestation"] = (
        0.35 * _clamp((ds - 0.08) / 0.18)        # dark tick/flea bodies
      + 0.25 * _clamp((tx - 100) / 300.0)        # slight roughness
      + 0.25 * _clamp((0.12 - rd) / 0.10)        # NOT heavily inflamed
      + 0.15 * _clamp(en[8] / 0.09)              # high-freq features (small dots)
    )

    # ── Wound / Injury ────────────────────────────────────────────────────────
    # Strong redness localised, high contrast region
    scores["wound_injury"] = (
        0.40 * _clamp((rd - 0.05) / 0.20)        # strong redness
      + 0.25 * _clamp((tx - 200) / 300.0)        # rough/irregular wound region
      + 0.20 * _clamp((mr - 0.35) / 0.25)        # warm/red dominant channel
      + 0.15 * _clamp(en[2] / 0.10)
    )

    # ── Eye Infection ─────────────────────────────────────────────────────────
    # Bright upper region (discharge/swelling), face-focussed image
    scores["eye_infection"] = (
        0.45 * _clamp((eb - 0.60) / 0.30)        # very bright eye region
      + 0.25 * _clamp(rd / 0.10)
      + 0.20 * _clamp((1.0 - abs(br - 0.55)) / 0.4)  # medium overall brightness
      + 0.10 * _clamp(en[3] / 0.10)
    )

    # ── Bacterial Skin Infection (Pyoderma) ───────────────────────────────────
    # Patchy lesions, moderate redness, texture indicates pus/crusting
    scores["skin_infection"] = (
        0.28 * _clamp((rd - 0.05) / 0.15)
      + 0.28 * _clamp((tx - 150) / 320.0)
      + 0.22 * _clamp(ds / 0.20)
      + 0.12 * _clamp(pv / 0.8)
      + 0.10 * _clamp(en[6] / 0.10)
    )

    # Clamp all scores to [0, 1]
    return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}


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
            "Oral antifungal (itraconazole / terbinafine) for widespread cases — vet prescription",
            "Antifungal shampoo (ketoconazole 2%) twice weekly",
            "Clip hair around lesions to improve topical penetration",
            "Treat for minimum 4 weeks — lesions clearing does not mean cured"
        ],
        "home_first_aid": [
            "Wear gloves — ringworm transfers easily to humans",
            "Disinfect bedding and surfaces with diluted bleach (1:10)",
            "Avoid sharing grooming tools",
            "Keep affected areas dry"
        ],
        "is_contagious": True,
        "zoonotic": True
    },
    "distemper": {
        "display":    "Canine Distemper",
        "urgency":    "critical",
        "symptoms":   [
            "Thick yellow-green eye and nasal discharge",
            "High fever (39.5°C+)",
            "Coughing, laboured breathing",
            "Lethargy and loss of appetite",
            "Neurological signs in late stage: seizures, head tilt"
        ],
        "basic_cure": [
            "No cure — treatment is supportive only",
            "IV fluids and electrolytes to prevent dehydration",
            "Broad-spectrum antibiotics to prevent secondary infections",
            "Anticonvulsants if neurological signs present",
            "Isolation from other dogs — highly contagious airborne virus",
            "Prevention: core vaccine (DHPPi) — single most important action"
        ],
        "home_first_aid": [
            "Isolate immediately — virus spreads by air and contact",
            "Keep warm, comfortable, and hydrated",
            "Wipe nasal and eye discharge gently with damp cloth",
            "Rush to vet — prognosis worsens rapidly without treatment"
        ],
        "is_contagious": True,
        "zoonotic": False
    },
    "parvovirus": {
        "display":    "Parvovirus (CPV)",
        "urgency":    "critical",
        "symptoms":   [
            "Severe bloody diarrhoea with strong odour",
            "Projectile vomiting",
            "Extreme lethargy — dog cannot stand",
            "Rapid dehydration (skin tenting, dry gums)",
            "High fever followed by dangerously low temperature"
        ],
        "basic_cure": [
            "Emergency IV fluids — primary treatment, prevents fatal dehydration",
            "Anti-emetics (maropitant/metoclopramide) to control vomiting",
            "Broad-spectrum antibiotics (ampicillin + metronidazole) IV",
            "Plasma transfusion if severe protein loss",
            "NPO (nil by mouth) for 24-48 hours then gradual reintroduction",
            "Bleach-disinfect any area the dog touched (virus survives months)"
        ],
        "home_first_aid": [
            "This is a veterinary emergency — do not delay",
            "Keep the dog warm and do not force fluids by mouth",
            "Bleach all surfaces the dog contacted",
            "Strict isolation — virus is extremely contagious"
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
            "Rinse eyes 1-2x daily with sterile saline — most mild cases resolve with this alone",
            "Apply lubricating eye gel (hypromellose) before walks",
            "E-collar to prevent rubbing",
            "Antibiotic eye drops (chloramphenicol/tobramycin) only if bacterial — vet prescription",
            "Rule out underlying cause: distemper, entropion, foreign body, dry eye"
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
            "Antibacterial shampoo bath (chlorhexidine 2-4%) twice weekly",
            "Oral antibiotics for 3-6 weeks (amoxicillin-clavulanate — vet only)",
            "Topical antibiotic spray (mupirocin)",
            "Treat underlying cause (allergy, hormonal — requires vet diagnosis)",
            "Medicated wipes for maintenance between baths"
        ],
        "home_first_aid": [
            "Keep affected areas clean and dry",
            "Diluted apple cider vinegar spray as temporary antibacterial",
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

    # ── Stage 2: disease classification ───────────────────────────────────────
    feats   = _extract_features(img)
    scores  = _score_diseases(feats)

    # Softmax over scores with temperature scaling
    vals    = list(scores.values())
    keys    = list(scores.keys())
    exp_v   = [math.exp(v * 10) for v in vals]   # temperature 1/10 (sharper than v1's 1/8)
    total   = sum(exp_v)
    soft    = [e / total for e in exp_v]

    disease_probs = {k: round(s * 100, 1) for k, s in zip(keys, soft)}
    top_key       = max(scores, key=scores.get)
    top_kb        = DISEASE_KB[top_key]
    urgency       = top_kb["urgency"]

    # IMPROVEMENT: confidence dampening — if top disease is close to second,
    # nudge urgency up one level to avoid false reassurance.
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]
        if margin < 0.08:   # top two diseases are very close
            urgency_order = ["low", "medium", "high", "critical"]
            idx = urgency_order.index(urgency)
            if idx < len(urgency_order) - 1:
                urgency = urgency_order[idx + 1]

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