import os
import sys
import uuid

sys.dont_write_bytecode = True
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename
from disease_classifier import predict
from ngo_locator import get_contacts_by_coords, get_contacts_by_city, get_all_cities

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT    = {"png", "jpg", "jpeg", "jfif", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif"}
# FIX: raised from 25 MB to 30 MB — mobile cameras (especially HEIC) can hit 20+ MB
MAX_MB         = 30

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# FIX: Add CORS-friendly headers + no-cache so mobile browsers don't serve
# a stale cached response as "server could not read the response"
@app.after_request
def add_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def allowed(filename):
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_uploaded_image(file):
    original = secure_filename(file.filename or "")
    if original and "." in original and not allowed(original):
        raise ValueError("Unsupported image type")

    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        # FIX: cap at 1024×1024 — same as before but also strip all EXIF/metadata
        # which can cause PIL to barf on some Samsung/Xiaomi HEIC files
        img.thumbnail((1024, 1024))
    except UnidentifiedImageError:
        raise ValueError("Could not read this image. Try JPG, PNG, WEBP, HEIC, BMP or TIFF.")
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    img.save(fpath, "JPEG", quality=88, optimize=True)
    return fname, fpath


@app.route("/")
def index():
    cities = get_all_cities()
    return render_template("index.html", cities=cities)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file received"}), 400

    try:
        fname, fpath = save_uploaded_image(file)
        result = predict(fpath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except MemoryError:
        return jsonify({"error": "Image too large to process. Please try a smaller photo."}), 400
    except Exception as e:
        app.logger.error(f"/predict error: {e}", exc_info=True)
        return jsonify({"error": "Analysis failed. Please try another image."}), 500

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
            app.logger.error(f"/ngo coords error: {e}", exc_info=True)
            return jsonify({"error": "Could not look up NGOs. Please try entering a city name."}), 500
    elif city:
        try:
            result = get_contacts_by_city(city)
        except Exception as e:
            app.logger.error(f"/ngo city error: {e}", exc_info=True)
            return jsonify({"error": "Could not look up NGOs for that city."}), 500
    else:
        return jsonify({"error": "Provide lat/lon or city"}), 400

    return jsonify(result)


@app.route("/cities")
def cities_route():
    return jsonify(get_all_cities())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)