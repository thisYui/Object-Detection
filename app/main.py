from pathlib import Path
from uuid import uuid4
import os

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

try:
    from .predict import detect_image
except ImportError:
    from predict import detect_image

from src.inference.predict_faster_rcnn import DEFAULT_CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "static" / "uploads"
RESULT_DIR = APP_ROOT / "static" / "results"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "faster_rcnn" / "best.pth"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
DEFAULT_THRESHOLD = 0.5


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "object-detection-dev")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(413)
    def file_too_large(_error):
        flash("The file is too large. Please choose an image under 16 MB.", "error")
        return redirect(url_for("index"))

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            threshold=DEFAULT_THRESHOLD,
            model_name="Faster R-CNN",
            weights_path=WEIGHTS_PATH,
            class_names=DEFAULT_CLASS_NAMES,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        upload = request.files.get("image")
        threshold = parse_threshold(request.form.get("threshold"))

        if upload is None or upload.filename == "":
            flash("Please choose an image before running detection.", "error")
            return redirect(url_for("index"))

        if not allowed_file(upload.filename):
            flash("Unsupported image format. Please use JPG, PNG, BMP, or WEBP.", "error")
            return redirect(url_for("index"))

        if not WEIGHTS_PATH.exists():
            flash(f"Model weights were not found: {WEIGHTS_PATH}", "error")
            return redirect(url_for("index"))

        original_name = secure_filename(upload.filename)
        suffix = Path(original_name).suffix.lower()
        stem = Path(original_name).stem or "image"
        unique_name = f"{stem}_{uuid4().hex[:10]}{suffix}"
        result_name = f"{Path(unique_name).stem}_result.jpg"

        upload_path = UPLOAD_DIR / unique_name
        result_path = RESULT_DIR / result_name
        upload.save(upload_path)

        try:
            prediction = detect_image(
                image_path=upload_path,
                output_path=result_path,
                weights_path=WEIGHTS_PATH,
                threshold=threshold,
            )
        except Exception as exc:
            flash(f"Could not run inference: {exc}", "error")
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            model_name="Faster R-CNN",
            threshold=threshold,
            detections=prediction["detections"],
            detection_count=len(prediction["detections"]),
            inference_seconds=prediction["inference_seconds"],
            device=prediction["device"],
            upload_url=url_for("static", filename=f"uploads/{unique_name}"),
            result_url=url_for("static", filename=f"results/{result_name}"),
            upload_filename=original_name,
            class_names=DEFAULT_CLASS_NAMES,
        )

    @app.route("/api/webcam-detect", methods=["POST"])
    def webcam_detect():
        upload = request.files.get("image")
        threshold = parse_threshold(request.form.get("threshold"))

        if upload is None or upload.filename == "":
            return jsonify({"error": "Missing image frame."}), 400

        if not WEIGHTS_PATH.exists():
            return jsonify({"error": f"Missing weights: {WEIGHTS_PATH}"}), 500

        unique_name = f"webcam_{uuid4().hex[:10]}.jpg"
        result_name = f"{Path(unique_name).stem}_result.jpg"
        upload_path = UPLOAD_DIR / unique_name
        result_path = RESULT_DIR / result_name
        upload.save(upload_path)

        try:
            prediction = detect_image(
                image_path=upload_path,
                output_path=result_path,
                weights_path=WEIGHTS_PATH,
                threshold=threshold,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(
            {
                "detections": prediction["detections"],
                "detection_count": len(prediction["detections"]),
                "threshold": threshold,
                "device": prediction["device"],
                "inference_seconds": prediction["inference_seconds"],
            }
        )

    return app


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(max(threshold, 0.05), 0.95)


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
