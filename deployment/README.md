# deployment/ — Containerization & cloud deploy (AASD 4016)

How to run and deploy the Dental Cavity Detector serving API. Tickets: **SCRUM-43** (Docker) · **SCRUM-44** (cloud).

## What's here / how it fits

- The container is defined by **`api/Dockerfile`** (+ `api/.dockerignore`). It runs the Flask app under **gunicorn** (production server) and reads the port from `$PORT`.
- The fine-tuned model weights download automatically from Hugging Face on the **first** `/predict` call (≈5 MB), so no weights are baked into the image. (Optional: bake them in at build time for a faster cold start — see note below.)

## 1. Build & run the container locally (Docker)

```bash
cd api
docker build -t dental-cavity-api .
docker run -p 8000:8000 dental-cavity-api
# then, in another terminal:
curl http://localhost:8000/health
curl -F "image=@sample_xray.png" http://localhost:8000/predict
```

## 2. Deploy to the cloud

> ⚠️ **The final deploy needs a cloud account + login**, which is the one step that must be done by a team member (Ali/Hessam) — an AI agent can't create accounts, enter credentials, or add a payment method. Everything up to that point is ready.

**Memory note:** PyTorch + ultralytics need roughly **1–2 GB RAM** at inference. Pick an instance/plan with **≥ 1 GB** or the first prediction may be killed (OOM). A 512 MB free tier is fine for `/health` but likely too small for `/predict`.

### Option A — Render (simplest: deploys straight from GitHub)

1. Sign in at render.com and click **New → Web Service**.
2. Connect the `neuroarcane/dental-cavity-detector` repo.
3. Set **Root Directory** = `api` (Render auto-detects the Dockerfile).
4. Choose an instance with **≥ 1 GB RAM** (the free 512 MB tier may OOM on `/predict`).
5. Health check path: `/health`. Create the service — Render builds the image and gives you a public URL.

A ready-made blueprint is committed at the repo root (`render.yaml`) if you prefer Render's Blueprint flow.

### Option B — Google Cloud Run (scales to zero, configurable memory, generous free tier)

Requires a Google Cloud account with billing enabled (free credits usually cover a demo).

```bash
gcloud run deploy dental-cavity-api \
  --source api \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300 \
  --allow-unauthenticated
```

Cloud Run builds the container from `api/Dockerfile`, deploys it, and returns a public HTTPS URL.

## 3. After deploy — verify

```bash
curl https://YOUR-PUBLIC-URL/health
curl -F "image=@sample_xray.png" https://YOUR-PUBLIC-URL/predict
```

When the public URL is live, update the Full Stack report (§7), the presentation draft (Slide 8/10), and move **SCRUM-44** to Done.

## Notes

- **Optional — bake weights into the image** (faster cold start): add a build step that runs `huggingface_hub.hf_hub_download(...)` so the `.pt` is inside the image; then the first request doesn't wait on a download.
- **Production hardening** (auth, HTTPS, rate limiting, etc.) is tracked separately — see §8 of the Full Stack report. Cloud Run/Render give HTTPS out of the box.

_Placeholder replaced 5 Aug 2026 (SCRUM-43 done; SCRUM-44 ready to deploy pending account credentials)._
