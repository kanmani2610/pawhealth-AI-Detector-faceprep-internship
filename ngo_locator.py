"""
ngo_locator.py - Find nearby vets and animal NGOs using OpenStreetMap.

Fallback: if Overpass is unreachable, serve contacts from the local static DB.
"""

import json, math, os, urllib.request, urllib.parse, urllib.error

# ── Static DB fallback (unchanged from original) ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "ngo_contacts.json")
_db     = None

# ── Overpass API endpoint (multiple mirrors for reliability) ─────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

_RADIUS_METERS  = 10000   # 10 km search radius
_MAX_RESULTS    = 6
_TIMEOUT_SECS   = 8


# ── Helpers ──────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R   = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _load_db():
    global _db
    try:
        with open(DB_PATH, "r") as f:
            _db = json.load(f)
    except Exception as e:
        print(f"[ngo_locator] Static DB load failed: {e}")
        _db = {"cities": {}, "national_helpline": {"name": "Animal Welfare Board", "phone": "1800-200-0167"}}


def _get_national():
    if _db is None:
        _load_db()
    return _db.get("national_helpline", {"name": "Animal Welfare Board of India", "phone": "1800-200-0167"})


# ── Overpass query builder ────────────────────────────────────────────────────

def _build_overpass_query(lat: float, lon: float, radius: int) -> str:
    """
    Queries OSM for:
      - amenity=veterinary       (vet clinics)
      - amenity=animal_shelter   (animal NGOs/shelters)
      - amenity=animal_boarding  (boarding that often has vets)
    Returns both nodes and ways (some clinics are mapped as buildings).
    """
    return f"""
[out:json][timeout:10];
(
  node["amenity"="veterinary"](around:{radius},{lat},{lon});
  way["amenity"="veterinary"](around:{radius},{lat},{lon});
  node["amenity"="animal_shelter"](around:{radius},{lat},{lon});
  way["amenity"="animal_shelter"](around:{radius},{lat},{lon});
  node["amenity"="animal_hospital"](around:{radius},{lat},{lon});
  way["amenity"="animal_hospital"](around:{radius},{lat},{lon});
);
out center tags;
""".strip()


def _query_overpass(lat: float, lon: float) -> list:
    """Try each mirror until one responds. Returns raw OSM elements list."""
    query = _build_overpass_query(lat, lon, _RADIUS_METERS)
    data  = urllib.parse.urlencode({"data": query}).encode()

    for endpoint in _OVERPASS_ENDPOINTS:
        try:
            req  = urllib.request.Request(
                endpoint,
                data=data,
                headers={"User-Agent": "PawHealthAI/1.0 (stray dog welfare project)"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECS) as resp:
                result = json.loads(resp.read())
                elements = result.get("elements", [])
                print(f"[ngo_locator] Overpass OK via {endpoint} — {len(elements)} results")
                return elements
        except Exception as e:
            print(f"[ngo_locator] Overpass mirror failed ({endpoint}): {e}")

    return []   # all mirrors failed → caller will use static fallback


# ── OSM element → contact dict ────────────────────────────────────────────────

def _osm_to_contact(el: dict, user_lat: float, user_lon: float) -> dict:
    """Normalise an OSM node/way into our standard contact shape."""
    tags = el.get("tags", {})

    # Coordinates: nodes have lat/lon directly; ways have a 'center' key
    if el["type"] == "node":
        elat, elon = el.get("lat", user_lat), el.get("lon", user_lon)
    else:
        center = el.get("center", {})
        elat   = center.get("lat", user_lat)
        elon   = center.get("lon", user_lon)

    dist_km = round(_haversine(user_lat, user_lon, elat, elon), 1)

    # Amenity → type mapping
    amenity = tags.get("amenity", "veterinary")
    if "shelter" in amenity:
        ctype = "ngo"
    else:
        ctype = "vet"

    # Build a human-readable address from OSM addr tags
    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:suburb", ""),
        tags.get("addr:city", ""),
    ]
    address = ", ".join(p for p in addr_parts if p).strip(", ") or tags.get("addr:full", "")

    # Google Maps link using coordinates (always works, no key needed)
    maps_url = f"https://www.google.com/maps?q={elat},{elon}"

    # OSM sometimes has phone/website
    phone   = tags.get("phone") or tags.get("contact:phone")
    website = tags.get("website") or tags.get("contact:website")
    if website and not website.startswith("http"):
        website = "https://" + website

    name = tags.get("name") or tags.get("name:en") or "Veterinary Clinic"

    return {
        "name":          name,
        "type":          ctype,
        "emergency":     tags.get("emergency") == "yes",
        "address":       address,
        "phone":         phone,
        "email":         tags.get("email") or tags.get("contact:email"),
        "website":       website,
        "maps_url":      maps_url,
        "distance_km":   dist_km,
        "rating":        None,   # OSM doesn't have ratings
        "total_ratings": None,
        "open_label":    None,   # OSM opening_hours parsing out of scope
        "source":        "osm",
    }


# ── Main OSM lookup ───────────────────────────────────────────────────────────

def _osm_vets_by_coords(lat: float, lon: float) -> dict:
    elements = _query_overpass(lat, lon)

    if not elements:
        return None   # signal to caller: fall through to static DB

    contacts = [_osm_to_contact(el, lat, lon) for el in elements]

    # Sort by distance, deduplicate by name+distance
    seen     = set()
    unique   = []
    for c in sorted(contacts, key=lambda x: x["distance_km"]):
        key = (c["name"].lower().strip(), c["distance_km"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return {
        "found":    len(unique) > 0,
        "city":     "Nearby",
        "contacts": unique[:_MAX_RESULTS],
        "source":   "osm",
        "national": _get_national(),
    }


# ── Geocoding (city name → lat/lon) via Nominatim (free, OSM-based) ──────────

def _geocode_city(city_name: str):
    """Use OSM Nominatim to turn a city name into lat/lon. No key needed."""
    params = urllib.parse.urlencode({
        "q":              city_name + ", India",
        "format":         "json",
        "limit":          "1",
        "addressdetails": "0",
    })
    url = "https://nominatim.openstreetmap.org/search?" + params
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PawHealthAI/1.0 (stray dog welfare project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECS) as resp:
            results = json.loads(resp.read())
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        print(f"[ngo_locator] Nominatim geocode failed for '{city_name}': {e}")
    return None, None


# ── Static DB helpers (unchanged logic from original) ────────────────────────

def _static_by_coords(lat: float, lon: float, max_results: int = 5) -> dict:
    if _db is None:
        _load_db()
    best_city = None
    best_dist = float("inf")
    for city_key, city_data in _db["cities"].items():
        d = _haversine(lat, lon, city_data["lat"], city_data["lon"])
        if d < best_dist:
            best_dist = d
            best_city = city_key
    if best_city is None or best_dist > 300:
        return {
            "found":       False,
            "city":        "Unknown",
            "distance_km": round(best_dist, 1),
            "contacts":    [],
            "source":      "static",
            "national":    _db["national_helpline"],
        }
    contacts = sorted(
        _db["cities"][best_city]["contacts"],
        key=lambda c: (0 if c["emergency"] else 1, 0 if c["type"] == "ngo" else 1)
    )
    return {
        "found":       True,
        "city":        best_city.title(),
        "distance_km": round(best_dist, 1),
        "contacts":    contacts[:max_results],
        "source":      "static",
        "national":    _db["national_helpline"],
    }


def _static_by_city(city_name: str) -> dict:
    if _db is None:
        _load_db()
    key = city_name.strip().lower()
    if key in _db["cities"]:
        return {
            "found":    True,
            "city":     key.title(),
            "contacts": _db["cities"][key]["contacts"],
            "source":   "static",
            "national": _db["national_helpline"],
        }
    return {"found": False, "city": city_name, "contacts": [], "source": "static", "national": _db["national_helpline"]}


# ── Public API (same signatures as before — nothing else needs to change) ─────

def get_contacts_by_coords(lat: float, lon: float, max_results: int = 5) -> dict:
    """GPS-based lookup. Tries Overpass/OSM first, falls back to static DB."""
    static = _static_by_coords(lat, lon, max_results)
    if static.get("found"):
        return static

    result = _osm_vets_by_coords(lat, lon)
    if result is not None:
        return result
    print("[ngo_locator] Overpass failed — using static DB fallback")
    return static


def get_contacts_by_city(city_name: str) -> dict:
    """City-name lookup. Geocodes via Nominatim → Overpass, falls back to static DB."""
    static = _static_by_city(city_name)
    if static.get("found"):
        return static

    lat, lon = _geocode_city(city_name)
    if lat is not None:
        result = _osm_vets_by_coords(lat, lon)
        if result is not None:
            result["city"] = city_name.title()
            return result
    print(f"[ngo_locator] OSM path failed for '{city_name}' — using static DB fallback")
    return static


def get_all_cities() -> list:
    """Returns static DB city list (used for autocomplete datalist in HTML)."""
    if _db is None:
        _load_db()
    return sorted(_db["cities"].keys())


if __name__ == "__main__":
    print("Testing Overpass lookup for Chennai (13.08, 80.27)...")
    r = get_contacts_by_coords(13.08, 80.27)
    print(f"Source: {r['source']} | Found: {r['found']} | Count: {len(r['contacts'])}")
    for c in r["contacts"]:
        print(f"  [{c['type'].upper()}] {c['name']} — {c['distance_km']} km — {c.get('phone') or c['maps_url']}")
