from __future__ import annotations

import io
import math
import os
import random
import re
import json
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image

LOCAL_IMAGE_SIZE = 1024


# ── Exceptions ─────────────────────────────────────────────────────────────────

class NoCoverageError(Exception):
    """Raised when no curated AOI bbox contains the requested (lat, lon)."""


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_AOIS_JSON = os.path.join(_BASE_DIR, "custom_aois.json")
_UPLOAD_IMG_DIR = os.path.join(_BASE_DIR, "custom_images")

with open(_CUSTOM_AOIS_JSON,"r", encoding="utf-8") as f:
    CUSTOM_AOIS: dict[str,dict[str, Any]] = json.load(f)

# ── Geo helpers ────────────────────────────────────────────────────────────────

def compute_bbox(lat: float, lon: float, aoi_m: float):

    half_lat = aoi_m / 2.0 / 111320.0
    half_lon = aoi_m / 2.0 / (111320.0 * math.cos(math.radians(lat)))
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def resolve_aoi_by_coords(lat: float, lon: float) -> str:
    matches: list[str] = []
    for aoi_id, entry in CUSTOM_AOIS.items():
        lon_min, lat_min, lon_max, lat_max = compute_bbox(
            entry["lat"], entry["lon"], entry["aoi_m"]
        )
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            matches.append(aoi_id)
    if not matches:
        raise NoCoverageError(
            f"No curated AOI covers (lat={lat}, lon={lon})."
        )
    return random.choice(matches)


# ── Catalog views ──────────────────────────────────────────────────────────────

def list_aois_flat():
    return [
        {
            "id": aoi_id,
            "name": entry["name"],
            "il": entry["il"],
            "ilce": entry["ilce"],
            "mahalle": entry["mahalle"],
            "lat": entry["lat"],
            "lon": entry["lon"],
            "aoi_m": entry["aoi_m"],
            "pre_date": entry["pre_date"],
            "post_date": entry["post_date"],
        }
        for aoi_id, entry in CUSTOM_AOIS.items()
    ]


def get_hierarchy():
    hierarchy: dict[str, dict[str, list[str | None]]] = {}
    for entry in CUSTOM_AOIS.values():
        il_bucket = hierarchy.setdefault(entry["il"], {})
        mahalleler = il_bucket.setdefault(entry["ilce"], [])
        if entry["mahalle"] not in mahalleler:
            mahalleler.append(entry["mahalle"])
    return hierarchy


def get_aoi(aoi_id: str):
    if aoi_id not in CUSTOM_AOIS:
        raise KeyError(aoi_id)
    entry = CUSTOM_AOIS[aoi_id]
    return {
        "id": aoi_id,
        "name": entry["name"],
        "il": entry["il"],
        "ilce": entry["ilce"],
        "mahalle": entry["mahalle"],
        "lat": entry["lat"],
        "lon": entry["lon"],
        "aoi_m": entry["aoi_m"],
        "pre_date": entry["pre_date"],
        "post_date": entry["post_date"],
    }


# ── Image fetch ────────────────────────────────────────────────────────────────

def fetch_custom_pair(aoi_id: str,):
    if aoi_id not in CUSTOM_AOIS:
        raise KeyError(f"Unknown AOI id '{aoi_id}'.")

    entry = CUSTOM_AOIS[aoi_id]

    pre_img  = _load_rgb(_resolve(entry["before_image"]))
    post_img = _load_rgb(_resolve(entry["after_image"]))

    meta: dict[str, Any] = {
        "id":        aoi_id,
        "name":      entry["name"],
        "il":        entry["il"],
        "ilce":      entry["ilce"],
        "mahalle":   entry["mahalle"],
        "lat":       entry["lat"],
        "lon":       entry["lon"],
        "aoi_m":     entry["aoi_m"],
        "size":      LOCAL_IMAGE_SIZE,
        "pre_date":  entry["pre_date"],
        "post_date": entry["post_date"],
    }
    return pre_img, post_img, meta


# ── Image helpers ──────────────────────────────────────────────────────────────

def _resolve(path: str):
    """Resolve a registry path relative to this file unless already absolute."""
    return path if os.path.isabs(path) else os.path.join(_BASE_DIR, path)


def _load_rgb(path: str):
    """Open an image, force RGB, and return uint8 image array without resizing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom AOI image not found: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# ── Upload helper ──────────────────────────────────────────────────────────────

# Turkish-letter-aware slug map; we don't need full Unicode normalisation here,
# just enough so a name like "Antakya Akevler" yields "antakya-akevler".
_TR_SLUG_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def _slugify(text: str):
    s = (text or "").translate(_TR_SLUG_MAP).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "aoi"


def _validate_iso_date(s: str, field: str):
    """Accept YYYY-MM-DD and raise ValueError with a UI-friendly message."""
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be in YYYY-MM-DD format (got '{s}').")


def save_uploaded_aoi(
    *,
    name: str,
    il: str,
    ilce: str,
    mahalle: str | None,
    lat: float,
    lon: float,
    aoi_m: int,
    pre_date: str,
    post_date: str,
    pre_bytes: bytes,
    post_bytes: bytes,
):
    _validate_iso_date(pre_date, "pre_date")
    _validate_iso_date(post_date, "post_date")

    base_slug = "-".join(filter(None, [
        _slugify(il), _slugify(ilce), _slugify(name)
    ]))
    aoi_id = f"{base_slug}-{uuid.uuid4().hex[:6]}"

    os.makedirs(_UPLOAD_IMG_DIR, exist_ok=True)
    pre_rel  = f"custom_images/upload_{aoi_id}_pre.png"
    post_rel = f"custom_images/upload_{aoi_id}_post.png"

    # Decode → force RGB → re-encode as PNG. Catches non-image uploads and
    # normalises whatever the user sent (JPEG, WebP, RGBA PNG, …).
    for rel, data, label in (
        (pre_rel,  pre_bytes,  "pre"),
        (post_rel, post_bytes, "post"),
    ):
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.convert("RGB").save(_resolve(rel), format="PNG")
        except Exception as e:
            raise ValueError(f"Could not read {label} image: {e}")

    entry: dict[str, Any] = {
        "name":      name,
        "il":        il,
        "ilce":      ilce,
        "mahalle":   mahalle,
        "lat":       float(lat),
        "lon":       float(lon),
        "aoi_m":     int(aoi_m),
        "pre_date":  pre_date,
        "post_date": post_date,
        # LEVIR-CD swap: newer file → before_image, older file → after_image.
        "before_image": post_rel,
        "after_image":  pre_rel,
    }

    CUSTOM_AOIS[aoi_id] = entry

    with open(_CUSTOM_AOIS_JSON, "w", encoding="utf-8") as f:
        json.dump(CUSTOM_AOIS, f, indent=2, ensure_ascii=False)

    return aoi_id
