# api/ — Model serving API (AASD 4016)

Flask API that serves predictions from the trained YOLO11n dental detector.

## Related Jira tickets
- **SCRUM-36** — Package the trained YOLO11 model for serving (Aparna)
- **SCRUM-37** — Build a Flask API that serves model predictions (Hessam)
- **SCRUM-46** — API-as-a-data-service: docs + tests (Hessam, with Aparna)

## Planned contents
- `app.py` — Flask app: POST an X-ray → JSON of boxes, labels, confidences; health check; input validation
- `model.py` — load weights, preprocess, run inference, return structured predictions
- `requirements.txt` — API dependencies
- `tests/` — endpoint tests

_Placeholder — replace this file as the API is built._
