"""
Dental Cavity Detector - Client App
-------------------------------------
Uploads a dental X-ray image to the deployed detection API and displays
the returned detections (bounding boxes + labels + confidence scores).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import os
import subprocess
from urllib.parse import urlsplit

import requests
import streamlit as st
from PIL import Image, ImageDraw

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest
    _HAS_GOOGLE_AUTH = True
except Exception:  # google-auth not installed (e.g. a minimal local run)
    _HAS_GOOGLE_AUTH = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def resolve_api_url() -> str:
    """Which API to call, in priority order.

    1. Streamlit secret CAVITY_API_URL  -> used by the public demo on
       Streamlit Community Cloud (set to our private Cloud Run /predict URL).
    2. Env var CAVITY_API_URL           -> handy for local overrides.
    3. Private Cloud Run default        -> our internal production endpoint.
    """
    try:
        if "CAVITY_API_URL" in st.secrets:
            return str(st.secrets["CAVITY_API_URL"])
    except Exception:
        pass
    return os.environ.get(
        "CAVITY_API_URL",
        "https://dental-cavity-api-158898993155.us-central1.run.app/predict",
    )

# Fixed, distinct colors per class so boxes/legend stay consistent and legible.
CLASS_COLORS = {
    "cavity": "#E24A4A",
    "filling": "#3E8EDE",
    "crown": "#E8A33D",
    "impacted tooth": "#8B5FBF",
}
FALLBACK_COLORS = ["#4CAF87", "#D45BB0", "#6E7CE0", "#C9A227"]

st.set_page_config(page_title="Dental Cavity Detector", page_icon="🦷", layout="centered")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_identity_token(audience: str) -> str:
    """Mint a Google-signed ID token to call the private Cloud Run service.

    Priority:
    1. Service-account key in Streamlit secrets ([gcp_service_account]) -- used
       by the public demo on Streamlit Community Cloud. The token's audience is
       the Cloud Run service base URL, and the service account must hold the
       Cloud Run Invoker role on that service.
    2. Local `gcloud` CLI (gcloud auth print-identity-token) for developers who
       have run `gcloud auth login` with Invoker access.
    """
    # 1. Service account supplied via Streamlit secrets (public demo).
    try:
        has_sa = "gcp_service_account" in st.secrets
    except Exception:
        has_sa = False

    if has_sa:
        if not _HAS_GOOGLE_AUTH:
            raise RuntimeError(
                "google-auth is required for service-account auth. "
                "Add 'google-auth' to requirements.txt."
            )
        info = dict(st.secrets["gcp_service_account"])
        creds = service_account.IDTokenCredentials.from_service_account_info(
            info, target_audience=audience,
        )
        creds.refresh(GoogleAuthRequest())
        return creds.token

    # 2. Local gcloud fallback.
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("gcloud returned an empty token.")
    return token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_color_for_label(label: str) -> str:
    key = label.strip().lower()
    if key in CLASS_COLORS:
        return CLASS_COLORS[key]
    idx = sum(ord(c) for c in key) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def normalize_detections(payload):
    """Normalize a few common API response shapes into a flat list of dicts."""
    if isinstance(payload, dict):
        items = payload.get("predictions") or payload.get("detections") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("class") or item.get("class_name") or "finding"
        confidence = item.get("confidence") or item.get("score") or 0.0
        box = item.get("box") or item.get("bbox") or item.get("bounding_box")

        if isinstance(box, dict):
            box = [
                box.get("x1", box.get("xmin", 0)),
                box.get("y1", box.get("ymin", 0)),
                box.get("x2", box.get("xmax", 0)),
                box.get("y2", box.get("ymax", 0)),
            ]
        if box is None or len(box) != 4:
            continue

        normalized.append({
            "label": str(label),
            "confidence": float(confidence),
            "box": [float(v) for v in box],
        })
    return normalized


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """
    Draw color-coded bounding boxes only (no inline text labels). With
    dozens of overlapping detections, per-box text labels collide and
    become unreadable -- a color legend + results table below communicates
    the same information far more cleanly.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Scale line width to image size so boxes stay visible on large X-rays.
    line_width = max(2, round(min(img.size) / 250))

    for det in sorted(detections, key=lambda d: -d["confidence"]):
        x1, y1, x2, y2 = det["box"]
        color = get_color_for_label(det["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    return img


def draw_legend(detections: list):
    """Render a compact color legend for the classes present in this result."""
    labels_present = sorted({d["label"] for d in detections})
    if not labels_present:
        return
    cols = st.columns(len(labels_present))
    for col, label in zip(cols, labels_present):
        color = get_color_for_label(label)
        col.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span style='width:14px;height:14px;border-radius:3px;background:{color};"
            f"display:inline-block;'></span>"
            f"<span style='font-size:0.9rem;'>{label}</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Minimal styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 900px;}
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.6rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - kept minimal for demo; advanced knobs tucked away
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🦷 Dental Cavity Detector")
    st.caption("AI-assisted X-ray findings")

    confidence_threshold = st.slider("Minimum confidence to show", 0.0, 1.0, 0.3, 0.05)

    with st.expander("Advanced settings"):
        api_url = st.text_input("API endpoint", value=resolve_api_url())
        timeout_seconds = st.slider("Request timeout (seconds)", 10, 180, 90, 10)
        use_auth = st.checkbox(
            "Send Google auth token",
            value=True,
            help="Keep ON: our Cloud Run endpoint is private, so each request "
                 "must carry a Google identity token (from the service account).",
        )
        field_name = "image"

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0;'>🦷 Dental Cavity Detector</h1>"
    "<p style='color:#6b7280;font-size:1.05rem;margin-top:0.3rem;'>"
    "Upload a dental X-ray to identify cavities, fillings, crowns, and impacted teeth.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an X-ray image", type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("⬆️ Upload an X-ray to get started.")
else:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    run_clicked = st.button("Analyze X-ray", type="primary")

    if not run_clicked:
        st.image(image, use_container_width=True)
    else:
        with st.spinner("Analyzing X-ray..."):
            headers = {}
            if use_auth:
                try:
                    parts = urlsplit(api_url)
                    audience = f"{parts.scheme}://{parts.netloc}"
                    headers["Authorization"] = f"Bearer {get_identity_token(audience)}"
                except Exception as e:
                    st.error(f"Could not authenticate with the API: {e}")
                    st.stop()

            try:
                files = {field_name: (uploaded_file.name, image_bytes, uploaded_file.type or "image/jpeg")}
                params = {"conf": confidence_threshold}
                response = requests.post(api_url, files=files, params=params, headers=headers, timeout=timeout_seconds)
                response.raise_for_status()
                payload = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the detection service. Please try again.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")
                st.stop()
            except requests.exceptions.HTTPError:
                st.error("The detection service returned an error. Please try again.")
                st.stop()
            except ValueError:
                st.error("Received an unexpected response from the detection service.")
                st.stop()

        detections = normalize_detections(payload)
        detections = [d for d in detections if d["confidence"] >= confidence_threshold]

        if not detections:
            st.image(image, use_container_width=True)
            st.info("No findings above the selected confidence threshold.")
        else:
            annotated = draw_detections(image, detections)
            st.image(annotated, use_container_width=True)

            st.markdown(f"**{len(detections)} finding(s) detected**")
            draw_legend(detections)
            st.write("")

            st.dataframe(
                [
                    {
                        "Finding": d["label"],
                        "Confidence": f"{d['confidence']*100:.0f}%",
                    }
                    for d in sorted(detections, key=lambda d: -d["confidence"])
                ],
                use_container_width=True,
                hide_index=True,
            )
