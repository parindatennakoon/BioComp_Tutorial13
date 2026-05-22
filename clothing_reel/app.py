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
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store: {job_id: {"status": ..., "progress": ..., "file": ..., "error": ...}}
jobs: dict[str, dict] = {}


def _run_job(job_id: str, url: str) -> None:
    try:
        jobs[job_id]["status"] = "scraping"
        jobs[job_id]["progress"] = 0

        products = scrape_products(url)
        if not products:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "No products found on that page."
            return

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Please provide a full URL starting with http:// or https://"}), 400

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "progress": 0, "total": 0, "file": None, "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, url), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename: str):
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
