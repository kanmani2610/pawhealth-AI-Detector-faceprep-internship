"""
ngo_locator.py - Find nearby vets and animal NGOs.

Serves contacts entirely from the local static DB (ngo_contacts.json).
No outbound network calls — works offline and behind any firewall/proxy.
"""

import json, math, os

# ── Static DB ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "ngo_contacts.json")
_db      = None


def _load_db():
    global _db
    try:
        with open(DB_PATH, "r") as f:
            _db = json.load(f)
    except Exception as e:
        print(f"[ngo_locator] Static DB load failed: {e}")
        _db = {"cities": {}, "national_helpline": {"name": "Animal Welfare Board", "phone": "1800-200-0167"}}


def _get_db():
    if _db is None:
        _load_db()
    return _db


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R  = 6371.0
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Public API ────────────────────────────────────────────────────────────────

def get_contacts_by_coords(lat: float, lon: float, max_results: int = 5) -> dict:
    """GPS-based lookup — finds nearest city in static DB."""
    db = _get_db()
    best_city = None
    best_dist = float("inf")
    for city_key, city_data in db["cities"].items():
        d = _haversine(lat, lon, city_data["lat"], city_data["lon"])
        if d < best_dist:
            best_dist = d
            best_city = city_key

    national = db.get("national_helpline", {"name": "Animal Welfare Board of India", "phone": "1800-200-0167"})

    if best_city is None or best_dist > 300:
        return {
            "found":       False,
            "city":        "Unknown",
            "distance_km": round(best_dist, 1) if best_dist != float("inf") else None,
            "contacts":    [],
            "source":      "static",
            "national":    national,
        }

    contacts = sorted(
        db["cities"][best_city]["contacts"],
        key=lambda c: (0 if c["emergency"] else 1, 0 if c["type"] == "ngo" else 1)
    )
    return {
        "found":       True,
        "city":        best_city.title(),
        "distance_km": round(best_dist, 1),
        "contacts":    contacts[:max_results],
        "source":      "static",
        "national":    national,
    }


def get_contacts_by_city(city_name: str) -> dict:
    """City-name lookup from static DB."""
    db  = _get_db()
    key = city_name.strip().lower()
    national = db.get("national_helpline", {"name": "Animal Welfare Board of India", "phone": "1800-200-0167"})

    # Exact match first
    if key in db["cities"]:
        return {
            "found":    True,
            "city":     key.title(),
            "contacts": db["cities"][key]["contacts"],
            "source":   "static",
            "national": national,
        }

    # Partial match (e.g. "bengaluru" → "bangalore")
    for city_key in db["cities"]:
        if key in city_key or city_key in key:
            return {
                "found":    True,
                "city":     city_key.title(),
                "contacts": db["cities"][city_key]["contacts"],
                "source":   "static",
                "national": national,
            }

    return {
        "found":    False,
        "city":     city_name.title(),
        "contacts": [],
        "source":   "static",
        "national": national,
    }


def get_all_cities() -> list:
    """Returns city list for autocomplete datalist in HTML."""
    return sorted(_get_db()["cities"].keys())


if __name__ == "__main__":
    print("Testing static lookup for Chennai (13.08, 80.27)...")
    r = get_contacts_by_coords(13.08, 80.27)
    print(f"Source: {r['source']} | Found: {r['found']} | Count: {len(r['contacts'])}")
    for c in r["contacts"]:
        print(f"  [{c['type'].upper()}] {c['name']} — {c.get('phone', 'N/A')}")