---
title: Satellite Change Detection
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Satellite Change Detection

Web app for **building change detection** on satellite imagery, built around a fine-tuned **ChangeFormerV6** model. Curated pre/post pairs cover earthquake-damage sites in Hatay and Kahramanmaraş (Feb 2023).

🌐 [Live demo](https://huggingface.co/spaces/omerfkk/change-detection) · 🧠 [Model weights](https://huggingface.co/omerfkk/changeformer-xbd)

## Stack

FastAPI · ChangeFormerV6 · Leaflet · Docker. Trained on LEVIR-CD ∪ LEVIR-CD+ ∪ xBD with two-stage focal-loss → OHEM cross-entropy fine-tuning.

## Run

```bash
docker build -t changeformer-app .
docker run -p 7860:7860 changeformer-app
```

Open `http://localhost:7860`. Weights are pulled from the Hub on first run.

## API

- `GET  /api/aois` — catalog
- `GET  /api/aois/lookup?lat=..&lon=..` — point → AOI
- `POST /api/aois` — upload custom pair
- `POST /api/detect` — `{ aoi_id | (lat, lon), threshold }`

## Notes

Final-year graduation project, Yeditepe University. `ChangeFormer/` is vendored from the original repo and retains its upstream license.
