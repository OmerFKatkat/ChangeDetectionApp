"""FastAPI app for the satellite change detection demo.

The app no longer pulls imagery from Maxar; every AOI is served from local
pre-saved PNG pairs registered in :mod:`custom_aois`. The frontend picks an
AOI either by browsing the il/ilçe/mahalle hierarchy or by submitting a
(lat, lon) the backend resolves against curated bboxes.
"""

import sys
import os
import io
import base64
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), 'ChangeFormer'))

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ChangeFormer.models.networks import define_G
from ChangeFormer.datasets.data_utils import CDDataAugmentation

from custom_aois import (
    fetch_custom_pair,
    list_aois_flat,
    get_hierarchy,
    get_aoi,
    resolve_aoi_by_coords,
    save_uploaded_aoi,
    NoCoverageError,
)

from huggingface_hub import hf_hub_download

# ── Model Loading ──────────────────────────────────────────────────────────────

class Args:
    pass

args = Args()
args.net_G     = "ChangeFormerV6"
args.embed_dim = 256
args.gpu_ids   = [0] if torch.cuda.is_available() else []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Loading ChangeFormer model...")
model = define_G(args=args, gpu_ids=args.gpu_ids)
ckp_path = hf_hub_download(
    repo_id="omerfkk/changeformer-xbd",
    filename="best_ckpt.pt",
    cache_dir="/tmp/hf_cache",
)
checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_G_state_dict'])
model.eval()
print("Model loaded successfully!")

augm = CDDataAugmentation(img_size=256)

# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="Satellite Change Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schemas ────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """Pick an AOI by id (preferred) or by coordinates (fallback).

    Exactly one of the two must be provided; the resolver in
    :func:`detect_changes` handles the dispatch and 400s on ambiguity.
    """

    aoi_id:    str | None   = Field(default=None, min_length=1)
    lat:       float | None = Field(default=None, ge=-90.0,  le=90.0)
    lon:       float | None = Field(default=None, ge=-180.0, le=180.0)
    threshold: float        = Field(0.5, ge=0.0, le=1.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def numpy_to_base64(img_array: np.ndarray, mode: str = "RGB") -> str:
    pil_img = Image.fromarray(img_array.astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_overlay(base_img: np.ndarray, strength_map: np.ndarray,
                   color=(255, 60, 60), alpha: float = 0.65) -> np.ndarray:
    overlay  = base_img.copy().astype(np.float32)
    strength = np.clip(strength_map.astype(np.float32), 0.0, 1.0)

    if strength.shape[:2] != base_img.shape[:2]:
        strength_pil = Image.fromarray((strength * 255).astype(np.uint8), mode="L")
        strength_pil = strength_pil.resize(
            (base_img.shape[1], base_img.shape[0]), Image.BILINEAR
        )
        strength = np.array(strength_pil).astype(np.float32) / 255.0

    color_layer = np.full_like(overlay, color, dtype=np.float32)
    alpha_map   = (strength[..., None] * alpha).clip(0.0, 1.0)
    overlay     = overlay * (1.0 - alpha_map) + color_layer * alpha_map
    return overlay.clip(0, 255).astype(np.uint8)


def preprocess(img1: np.ndarray, img2: np.ndarray):
    [t1, t2], _ = augm.transform([img1, img2], [], to_tensor=True)
    return t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device)


def run_inference(pre_img: np.ndarray, post_img: np.ndarray, threshold: float):
    t1, t2 = preprocess(pre_img, post_img)
    with torch.no_grad():
        outputs = model(t1, t2)
        pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        change_prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]
        prob_full = F.interpolate(
            change_prob, size=post_img.shape[:2],
            mode="bilinear", align_corners=False,
        )
    prob_map = prob_full.squeeze().cpu().numpy().astype(np.float32)
    binary = prob_map > threshold
    pred_mask = binary.astype(np.uint8)
    change_strength = prob_map * pred_mask
    return prob_map, pred_mask, change_strength


def _no_coverage_response() -> JSONResponse:
    """Shared 422 body for "no curated AOI covers this point". Used by both
    /api/detect (when no aoi_id is provided) and /api/aois/lookup."""
    return JSONResponse(
        {
            "found":   False,
            "success": False,
            "error":   "no_coverage",
            "message": (
                "Currently we don't have coverage at this location, but we "
                "may add more areas in the future."
            ),
        },
        status_code=422,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/aois")
def get_aois():
    """Full catalog plus the il/ilçe/mahalle hierarchy. Cheap — everything is
    computed in-process from CUSTOM_AOIS."""
    return JSONResponse({
        "aois":      list_aois_flat(),
        "hierarchy": get_hierarchy(),
    })


@app.get("/api/aois/lookup")
def lookup_aoi(lat: float, lon: float):
    if not (-90.0 <= lat <= 90.0):
        raise HTTPException(status_code=400, detail="lat must be in [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        raise HTTPException(status_code=400, detail="lon must be in [-180, 180].")

    try:
        aoi_id = resolve_aoi_by_coords(lat, lon)
    except NoCoverageError:
        return _no_coverage_response()

    return JSONResponse({"found": True, "aoi": get_aoi(aoi_id)})


# Max raw bytes accepted per uploaded image. 20 MB is generous for a 1024×1024
# PNG and still cheap to keep in memory while we re-encode it.
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.post("/api/aois")
async def upload_aoi(
    name:       str        = Form(..., min_length=1, max_length=120),
    il:         str        = Form(..., min_length=1, max_length=80),
    ilce:       str        = Form(..., min_length=1, max_length=80),
    mahalle:    str        = Form("",   max_length=80),
    lat:        float      = Form(..., ge=-90.0,  le=90.0),
    lon:        float      = Form(..., ge=-180.0, le=180.0),
    aoi_m:      int        = Form(400, ge=50, le=5000),
    pre_date:   str        = Form(..., min_length=10, max_length=10),   
    post_date:  str        = Form(..., min_length=10, max_length=10),
    pre_image:  UploadFile = File(...),
    post_image: UploadFile = File(...),
):
    """Append a new AOI to the in-memory catalog and persist custom_aois.json.

    Pre/post dates are kept truthful; the LEVIR-CD swap (newer file in the
    ``before_image`` slot) is applied internally by ``save_uploaded_aoi``.

    NOTE on persistence: on a Hugging Face Space the filesystem is wiped on
    restart, so uploads made here are session-only by default. To keep them,
    splice in ``huggingface_hub.HfApi().upload_file(...)`` calls inside
    ``save_uploaded_aoi``; see the marker comment there.
    """
    pre_bytes  = await pre_image.read()
    post_bytes = await post_image.read()

    if not pre_bytes or not post_bytes:
        raise HTTPException(status_code=400, detail="Both image files are required.")
    if len(pre_bytes) > _MAX_UPLOAD_BYTES or len(post_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
        )

    try:
        aoi_id = await asyncio.to_thread(
            save_uploaded_aoi,
            name=name.strip(),
            il=il.strip(),
            ilce=ilce.strip(),
            mahalle=(mahalle.strip() or None),
            lat=lat, lon=lon, aoi_m=aoi_m,
            pre_date=pre_date, post_date=post_date,
            pre_bytes=pre_bytes, post_bytes=post_bytes,
        )
    except ValueError as e:
        # Bad image data, bad date string, etc. — surface the message verbatim
        # so the frontend can show it in the modal.
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse({"success": True, "aoi": get_aoi(aoi_id)})


@app.post("/api/detect")
async def detect_changes(req: DetectRequest):
    """Resolve an AOI (by id, or by point if no id given), load its pre/post
    images, run ChangeFormer, return base64 PNGs + aggregate stats.
    """

    # ── Resolve target AOI ──────────────────────────────────────────────────
    if req.aoi_id:
        aoi_id = req.aoi_id
    elif req.lat is not None and req.lon is not None:
        try:
            aoi_id = resolve_aoi_by_coords(req.lat, req.lon)
        except NoCoverageError:
            return _no_coverage_response()
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either `aoi_id` or both `lat` and `lon`.",
        )

    # ── Load images ─────────────────────────────────────────────────────────
    try:
        pre_img, post_img, meta = await asyncio.to_thread(fetch_custom_pair, aoi_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown aoi_id '{aoi_id}'.")
    except FileNotFoundError as exc:
        # Bad registry entry — surface to logs/UI rather than silently 404.
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Inference + response ────────────────────────────────────────────────
    try:
        prob_map, pred_mask, change_strength = await asyncio.to_thread(
            run_inference, pre_img, post_img, req.threshold
        )

        total_pixels   = pred_mask.size
        changed_pixels = int(pred_mask.sum())
        change_pct     = round(changed_pixels / total_pixels * 100, 2)
        conf_vals      = prob_map[pred_mask == 1]
        avg_confidence = round(float(conf_vals.mean()), 4) if conf_vals.size else 0.0

        return JSONResponse({
            "success": True,
            "aoi_id":  aoi_id,
            "aoi": {
                "name":      meta["name"],
                "il":        meta["il"],
                "ilce":      meta["ilce"],
                "mahalle":   meta["mahalle"],
                "lat":       meta["lat"],
                "lon":       meta["lon"],
                "aoi_m":     meta["aoi_m"],
                "pre_date":  meta["pre_date"],
                "post_date": meta["post_date"],
            },
            "before":  numpy_to_base64(pre_img),
            "after":   numpy_to_base64(post_img),
            "mask":    numpy_to_base64(
                (change_strength * 255).astype(np.uint8), mode="L"
            ),
            "overlay": numpy_to_base64(create_overlay(post_img, change_strength)),
            "stats": {
                "change_percentage": change_pct,
                "confidence":        avg_confidence,
                "image_size":        f"{post_img.shape[0]}x{post_img.shape[1]}",
                "threshold":         req.threshold,
            },
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Serve Frontend ─────────────────────────────────────────────────────────────

app.mount(
    "/",
    StaticFiles(
        directory=os.path.join(os.path.dirname(__file__), "static"),
        html=True,
    ),
    name="static",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
