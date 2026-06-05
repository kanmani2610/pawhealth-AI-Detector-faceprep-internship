

import os
import sys
import uuid

sys.dont_write_bytecode = True
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from disease_classifier import predict
from ngo_locator import get_contacts_by_coords, get_contacts_by_city, get_all_cities

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT    = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_MB         = 10

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    cities = get_all_cities()
    return render_template("index.html", cities=cities)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not allowed(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    original = secure_filename(file.filename)
    ext      = original.rsplit(".", 1)[1].lower()
    fname    = f"{uuid.uuid4().hex}.{ext}"
    fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    try:
        result = predict(fpath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result["image_url"] = f"/static/uploads/{fname}"
    return jsonify(result)


@app.route("/ngo", methods=["POST"])
def ngo_route():
    data = request.get_json(silent=True) or {}
    lat  = data.get("lat")
    lon  = data.get("lon")
    city = data.get("city", "").strip()

    if lat is not None and lon is not None:
        try:
            result = get_contacts_by_coords(float(lat), float(lon))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif city:
        result = get_contacts_by_city(city)
    else:
        return jsonify({"error": "Provide lat/lon or city"}), 400

    return jsonify(result)


@app.route("/cities")
def cities_route():
    return jsonify(get_all_cities())


if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
