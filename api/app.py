"""
Dental Cavity Detector — model serving API (AASD 4016, SCRUM-37).

A small Flask app that serves predictions from the fine-tuned YOLO11n
dental detector. Upload an X-ray to POST /predict and get back the
detected findings (Cavity, Filling, Crown, Impacted Tooth) as JSON
bounding boxes with class labels and confidences.

Run:
    pip install -r requirements.txt
    python app.py                      # serves on http://127.0.0.1:8000

Weights: by default the fine-tuned model is pulled from Hugging Face
(aparnamohankumar/dental-cavity-detector). Override with a local file
via the DCD_WEIGHTS env var, e.g. DCD_WEIGHTS=/path/to/best.pt.
"""
import io
import os
import threading

from flask import Flask, request, jsonify

# ---- configuration -------------------------------------------------------
HF_REPO = os.getenv("DCD_HF_REPO", "aparnamohankumar/dental-cavity-detector")
HF_FILE = os.getenv("DCD_HF_FILE", "yolo11_baseline_best.pt")
LOCAL_WEIGHTS = os.getenv("DCD_WEIGHTS")          # optional local .pt override
MAX_MB = int(os.getenv("DCD_MAX_MB", "10"))       # reject uploads larger than this
DEFAULT_CONF = float(os.getenv("DCD_CONF", "0.25"))

app = Flask(__name__)

_model = None
_model_lock = threading.Lock()


def _weights_path():
    """Resolve the model weights: local override, else download from HF."""
    if LOCAL_WEIGHTS:
        return LOCAL_WEIGHTS
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)


def get_model():
    """Lazily load the YOLO model once, thread-safely."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO(_weights_path())
    return _model


def run_inference(image, conf=DEFAULT_CONF):
    """Run the detector on a PIL image, return a list of detections."""
    result = get_model().predict(image, conf=conf, verbose=False)[0]
    names = result.names
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0].tolist())
        detections.append({
            "label": names.get(int(box.cls[0]), str(int(box.cls[0]))),
            "confidence": round(float(box.conf[0]), 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return detections


# ---- routes --------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify(status="ok", model=f"{HF_REPO}/{HF_FILE}"), 200


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="missing file field 'image' (multipart/form-data)"), 400

    upload = request.files["image"]
    if not upload.filename:
        return jsonify(error="empty filename"), 400

    data = upload.read()
    if not data:
        return jsonify(error="empty file"), 400
    if len(data) > MAX_MB * 1024 * 1024:
        return jsonify(error=f"file too large (limit {MAX_MB} MB)"), 413

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify(error="uploaded file is not a valid image"), 400

    try:
        conf = float(request.args.get("conf", DEFAULT_CONF))
    except ValueError:
        return jsonify(error="'conf' must be a number between 0 and 1"), 400

    try:
        detections = run_inference(image, conf=conf)
    except Exception as exc:  # inference/model-load failure
        return jsonify(error=f"inference failed: {exc}"), 500

    return jsonify(count=len(detections), predictions=detections), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
