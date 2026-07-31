# api/ — Model serving API (AASD 4016)

Flask API that serves predictions from the fine-tuned **YOLO11n** dental detector
(Cavity, Filling, Crown, Impacted Tooth). Ticket: **SCRUM-37**.

## Run

```bash
pip install -r requirements.txt
python app.py            # serves on http://127.0.0.1:8000
```

By default the fine-tuned weights are pulled from Hugging Face
(`aparnamohankumar/dental-cavity-detector`). To use a local checkpoint instead:

```bash
DCD_WEIGHTS=/path/to/best.pt python app.py
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check → `{"status":"ok","model":"…"}` |
| POST | `/predict` | `multipart/form-data`, field `image`. Optional `?conf=0.25`. Returns JSON detections. |

### Example

```bash
curl -F "image=@sample_xray.png" "http://127.0.0.1:8000/predict?conf=0.25"
```

```json
{
  "count": 2,
  "predictions": [
    {"label": "Cavity", "confidence": 0.91, "box": {"x1": 10.0, "y1": 20.0, "x2": 60.0, "y2": 80.0}},
    {"label": "Crown",  "confidence": 0.88, "box": {"x1": 120.0, "y1": 40.0, "x2": 180.0, "y2": 110.0}}
  ]
}
```

Validation: missing/empty file → 400, non-image → 400, file over 10 MB → 413, inference error → 500.

## Tests

```bash
pytest tests/     # stubs the model; verifies routing, validation, and JSON shape
```

## Status & next steps

- **Done:** endpoints, input validation, JSON contract, unit tests (model stubbed) — all passing.
- **Final verification (Hessam):** run once with the real fine-tuned weights (first `/predict` downloads them from Hugging Face) and confirm a real X-ray returns dental detections.
- **Related tickets:** SCRUM-36 (package model for serving, Aparna) — the loader here can be swapped for that module when ready · SCRUM-43 (Dockerize) · SCRUM-44 (cloud deploy) · SCRUM-46 (API docs & tests).
