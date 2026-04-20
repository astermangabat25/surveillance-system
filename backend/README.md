# Local FastAPI backend scaffold

This folder is the local-first backend for the surveillance frontend.

## Recommended dev flow

1. Run frontend on Vercel/local with `npm run dev`
2. Run backend locally with FastAPI + Uvicorn
3. Keep runtime uploads/exports in `backend/storage/` and model weights in `backend/Occlusion-Robust-RTDETR/weights/models/`
4. Add database + Ultralytics integration after the UI is wired to real endpoints

## Suggested Python setup

Create a virtual environment and install the backend packages:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install "fastapi[standard]" python-multipart`

Run the API:

- `uvicorn backend.app.main:app --reload`

Docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Storage layout

- `backend/storage/videos/raw/` → uploaded source footage
- `backend/storage/videos/processed/` → rendered detections / tracked exports
- `backend/storage/exports/` → reports / downloadable artifacts

## Model weights path

- `backend/Occlusion-Robust-RTDETR/weights/models/` → uploaded active `.pt` / `.pth` model files
- `backend/storage/models/` remains as a legacy fallback lookup path for older deployments

## Required inference assets

- `backend/Occlusion-Robust-RTDETR/configs/` → RT-DETR config files
- `backend/Occlusion-Robust-RTDETR/inference_requirements/annotations/` → required annotations JSON
- `backend/Occlusion-Robust-RTDETR/inference_requirements/counting/` → counting line configs

These required files are intentionally kept outside `backend/storage/` so `backend/storage/` can be fully git-ignored as runtime data.

## API endpoints included

- `GET /api/locations`
- `POST /api/locations`
- `GET /api/videos`
- `GET /api/videos/{video_id}`
- `POST /api/videos`
- `GET /api/events`
- `GET /api/dashboard/summary`
- `GET /api/dashboard/traffic`
- `GET /api/dashboard/ai-synthesis`
- `GET /api/search?query=...`
- `GET /api/models/current`
- `GET /api/inference/status`
- `POST /api/models/upload`

## Ultralytics recommendation

Do **not** modify a random site-packages copy of Ultralytics.

If you must edit internals such as `block.py`, the safer path is:

1. choose the exact working Ultralytics tag,
2. fork/clone that version into a separate custom repo or vendor directory,
3. install it in editable mode for local/Vast development,
4. keep your model changes version-controlled.

Suggested local path for this project:

- `backend/vendor/ultralytics/`

That should be the step **after** the frontend is already talking to this API.
