import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()

from scraper import scrape_products
from video_generator import create_reel

app = Flask(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


def _restyle_products(products: list[dict], vibe: str, job_id: str) -> list[dict]:
    from ai_styler import restyle_image
    from video_generator import _download_image

    restyled = []
    for i, product in enumerate(products):
        try:
            local_path = product.get("local_image_path")
            if not local_path:
                img = _download_image(product.get("image_url", ""))
                if img is None:
                    restyled.append(product)
                    continue
                tmp_path = str(UPLOAD_DIR / f"tmp_{job_id}_{i}.jpg")
                img.save(tmp_path, "JPEG")
                local_path = tmp_path

            restyled_path = restyle_image(local_path, vibe)
            new_product = dict(product)
            new_product["local_image_path"] = restyled_path
            restyled.append(new_product)
        except Exception:
            restyled.append(product)  # fall back to original on error

    return restyled


def _run_job(job_id: str, url: str, vibe: str = "") -> None:
    try:
        jobs[job_id]["status"] = "scraping"
        products = scrape_products(url)
        if not products:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "No products found on that page."
            return

        if vibe:
            jobs[job_id]["status"] = "styling"
            jobs[job_id]["total"] = len(products)
            products = _restyle_products(products, vibe, job_id)

        jobs[job_id]["status"] = "generating"
        jobs[job_id]["total"] = len(products)

        filename = f"reel_{job_id}.mp4"
        output_path = str(OUTPUT_DIR / filename)

        def on_progress(i, total):
            jobs[job_id]["progress"] = i
            jobs[job_id]["total"] = total

        from urllib.parse import urlparse
        shop_name = urlparse(url).netloc

        create_reel(products, output_path, shop_name=shop_name, progress_callback=on_progress)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["file"] = filename
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


def _run_upload_job(job_id: str, products: list[dict], vibe: str, shop_name: str) -> None:
    try:
        if vibe:
            jobs[job_id]["status"] = "styling"
            jobs[job_id]["total"] = len(products)
            products = _restyle_products(products, vibe, job_id)

        jobs[job_id]["status"] = "generating"
        jobs[job_id]["total"] = len(products)

        filename = f"reel_{job_id}.mp4"
        output_path = str(OUTPUT_DIR / filename)

        def on_progress(i, total):
            jobs[job_id]["progress"] = i
            jobs[job_id]["total"] = total

        create_reel(products, output_path, shop_name=shop_name, progress_callback=on_progress)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["file"] = filename
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    vibe = (data.get("vibe") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Please provide a full URL starting with http:// or https://"}), 400

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "progress": 0, "total": 0, "file": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, url, vibe), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/generate-from-uploads", methods=["POST"])
def generate_from_uploads():
    count = int(request.form.get("count", 0))
    vibe = request.form.get("vibe", "").strip()
    shop_name = (request.form.get("shop_name") or "My Collection").strip()

    products = []
    job_id = uuid.uuid4().hex

    for i in range(count):
        file = request.files.get(f"image_{i}")
        name = request.form.get(f"name_{i}", "")
        price = request.form.get(f"price_{i}", "")
        if file:
            ext = Path(file.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            path = str(UPLOAD_DIR / f"upload_{job_id}_{i}{ext}")
            file.save(path)
            products.append({"name": name, "price": price, "local_image_path": path})

    if not products:
        return jsonify({"error": "No images uploaded"}), 400

    jobs[job_id] = {"status": "queued", "progress": 0, "total": 0, "file": None, "error": None}
    threading.Thread(target=_run_upload_job, args=(job_id, products, vibe, shop_name), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
