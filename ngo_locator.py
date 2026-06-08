

import json, math, os

DB_PATH = "data/ngo_contacts.json"
_db     = None


def _load_db():
    global _db
    with open(DB_PATH, "r") as f:
        _db = json.load(f)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    
    R   = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_contacts_by_city(city_name: str) -> dict:
    """Look up contacts by city name (case-insensitive)."""
    if _db is None:
        _load_db()
    key = city_name.strip().lower()
    if key in _db["cities"]:
        city = _db["cities"][key]
        return {
            "found":       True,
            "city":        key.title(),
            "contacts":    city["contacts"],
            "national":    _db["national_helpline"],
        }
    return {"found": False, "city": city_name, "contacts": [], "national": _db["national_helpline"]}


def get_contacts_by_coords(lat: float, lon: float, max_results: int = 5) -> dict:
    """Find nearest city from DB and return its contacts, sorted by type."""
    if _db is None:
        _load_db()

    best_city  = None
    best_dist  = float("inf")

    for city_key, city_data in _db["cities"].items():
        d = _haversine(lat, lon, city_data["lat"], city_data["lon"])
        if d < best_dist:
            best_dist  = d
            best_city  = city_key

    if best_city is None or best_dist > 300:
        # Too far from any known city — return national helpline only
        return {
            "found":       False,
            "city":        "Unknown",
            "distance_km": round(best_dist, 1),
            "contacts":    [],
            "national":    _db["national_helpline"],
        }

    city       = _db["cities"][best_city]
    contacts   = city["contacts"]

    # Sort: emergency first, then NGO, then clinics
    contacts_sorted = sorted(
        contacts,
        key=lambda c: (0 if c["emergency"] else 1, 0 if c["type"] == "ngo" else 1)
    )

    return {
        "found":       True,
        "city":        best_city.title(),
        "distance_km": round(best_dist, 1),
        "contacts":    contacts_sorted[:max_results],
        "national":    _db["national_helpline"],
    }


def get_all_cities() -> list:
    """Return list of all supported city names."""
    if _db is None:
        _load_db()
    return sorted(_db["cities"].keys())


if __name__ == "__main__":
    print("Supported cities:", get_all_cities())
    print("\nChennai contacts:")
    r = get_contacts_by_city("Chennai")
    for c in r["contacts"]:
        print(f"  [{c['type'].upper()}] {c['name']} — {c['phone']}")
    print("\nCoords test (13.0, 80.2 → should be Chennai):")
    r2 = get_contacts_by_coords(13.0, 80.2)
    print(f"  Nearest city: {r2['city']} ({r2['distance_km']} km)")