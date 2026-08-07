# dashboard/ — Stakeholder dashboard (AASD 4016)

Streamlit app that calls the deployed model API and visualizes predictions.

## Related Jira tickets

- **SCRUM-45** — Client app that calls the deployed API (Varsha)
- **SCRUM-51** — Dashboard connected to the live model (Varsha)

## Contents

- `app.py` — Streamlit app: upload an X-ray, calls the deployed endpoint, and renders
  color-coded bounding boxes, a confidence-filtered findings table, and a legend.
- `requirements.txt` — `streamlit`, `requests`, `pillow`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

- **API endpoint** — defaults to the GCP Cloud Run deployment; overridable via the
  `CAVITY_API_URL` env var or the "Advanced settings" panel in the sidebar.
- **Auth** — the Cloud Run service is private, so the app can send a Google-signed
  identity token (`gcloud auth print-identity-token`) with each request. Requires
  `gcloud auth login` once locally, with the Cloud Run Invoker role on the service.
  Toggle this off if pointing at an unauthenticated endpoint (e.g. a local server).

_Status: initial version working end-to-end against the deployed API (2026-08-06)._
