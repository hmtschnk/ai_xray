import argparse
from flask import Flask, render_template, request, jsonify, url_for, session, send_file, make_response, redirect
from torchxrayvision_APIv4_OO import ImagePreprocessor, XRayProcessor, XRayVisualizer
import os
import json
import uuid
import webbrowser
import matplotlib
matplotlib.use("Agg")   # Must come before importing pyplot or cm
import matplotlib.cm as cm
import threading
from datetime import datetime
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from safetensors import safe_open

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'  # Needed for session management and security

# Directories for input and output files
OUTFILE = 'static/outfile'
INFILE = 'static/infile'
os.makedirs(OUTFILE, exist_ok=True)  # Ensure output folder exists
os.makedirs(INFILE, exist_ok=True)   # Ensure input folder exists

# Predefined list of colormaps for overlays in visualization
PREDEFINED_COLORMAPS = [
    cm.jet, cm.viridis, cm.plasma, cm.inferno, cm.magma,
    cm.cividis, cm.cool, cm.hot, cm.spring, cm.summer
]

def get_thread_info():
    thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    return f"Thread-{thread_id} ({thread_name})"

@app.before_request
def log_request_thread():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}][Flask][{get_thread_info()}] Incoming request: {request.method} {request.path}")

@app.route('/')
def home():
    return "POST an image to /api/process or use /view/<id> to see results"

def get_image_url(image_id, selected_diseases):
    # Always point to dynamic overlay
    return url_for('render_overlay', image_id=image_id)

@app.route('/view/<image_id>')
def view(image_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}][Flask][{get_thread_info()}] View request for image ID: {image_id}")

    json_path = os.path.join(OUTFILE, f"{image_id}.json")
    if not os.path.exists(json_path):
        print(f"[{now}][Flask][{get_thread_info()}] JSON not found for ID {image_id}")
        return f"Data for ID {image_id} not found", 404

    with open(json_path, 'r') as file:
        data = json.load(file)

    diseases = data.get("results", {})
    input_img = data.get("input_img", "N/A")

    # Retrieve per-X-ray selection
    selected_diseases = session.get('selected_diseases_map', {}).get(image_id, [])
    image_url = get_image_url(image_id, selected_diseases)

    manual_list = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
                   'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
                   'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine']

    return render_template(
        'index_ori.html',
        image_url=image_url,
        diseases=diseases,
        input_img=input_img,
        xray_id=image_id,
        selected_diseases=selected_diseases,
        manual_list=manual_list
    )

@app.route('/api/process', methods=['POST'])
def process_upload():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.dcm', '.png']:
        return jsonify({"error": "Only .jpg and .dcm formats supported"}), 400

    unique_id = uuid.uuid4().hex
    input_path = os.path.join(INFILE, f"{unique_id}{ext}")
    file.save(input_path)

    try:
        processor = XRayProcessor(input_path, OUTFILE, unique_id)
        processed_id, result_data = processor.process()

        do_path = os.path.join(OUTFILE, f"DO-{unique_id}.safetensors")
        #ao_path = os.path.join(OUTFILE, f"AO-{unique_id}.safetensors")
        oo_path = os.path.join(OUTFILE, f"OO-{unique_id}.safetensors")
        missing_files = [p for p in (do_path, oo_path) if not os.path.isfile(p)]
        if missing_files:
            return jsonify({"error": f"Missing files: {', '.join(os.path.basename(f) for f in missing_files)}"}), 500

        result_data["id"] = unique_id
        result_data["input_img"] = file.filename
        view_url = url_for('view', image_id=unique_id, _external=True)

        webbrowser.open_new_tab(view_url)
        return jsonify({"view_url": view_url, "result": result_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/update_selected_diseases', methods=['POST'])
def update_selected_diseases():
    data = request.get_json()
    selected_diseases = data.get('selected_diseases', [])
    image_id = data.get('image_id', '')

    if 'selected_diseases_map' not in session:
        session['selected_diseases_map'] = {}

    session['selected_diseases_map'][image_id] = selected_diseases
    session.modified = True

    print(f"[INFO][{get_thread_info()}] Updated diseases for {image_id}: {selected_diseases}")

    overlay_url = url_for('render_overlay', image_id=image_id)
    cache_buster = int(datetime.utcnow().timestamp())
    return jsonify({"status": "success", "image_url": f"{overlay_url}?t={cache_buster}"})

@app.route('/render_overlay')
def render_overlay():
    image_id = request.args.get('image_id', '').split('?')[0].split('&')[0]
    selected_diseases = session.get('selected_diseases_map', {}).get(image_id, [])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f"[INFO][{get_thread_info()}] Rendering {image_id} with {selected_diseases}")

    try:
        alphas = [0.5] * len(selected_diseases)
        colormaps = [PREDEFINED_COLORMAPS[i % len(PREDEFINED_COLORMAPS)] for i in range(len(selected_diseases))]

        visualizer = XRayVisualizer(image_id, OUTFILE)
        image_io = visualizer.show_overlays(
            keys=selected_diseases,
            alphas=alphas,
            colormaps=colormaps,
            return_bytes=True
        )

        if image_io:
            response = make_response(send_file(image_io, mimetype='image/png'))
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response

        # No overlays: render OO safetensor as base image
        oo_path = os.path.join(OUTFILE, f"OO-{image_id}.safetensors")
        with safe_open(oo_path, framework="pt", device="cpu") as f:
            base_image = f.get_tensor("original_display_image").numpy()

        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)
        elif base_image.shape[-1] == 4:
            base_image = base_image[..., :3]
        base_image = np.clip(base_image, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(base_image)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        img_io = BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(img_io)
        plt.close(fig)
        img_io.seek(0)

        response = make_response(send_file(img_io, mimetype="image/png"))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    except Exception as e:
        print(f"[ERROR][{get_thread_info()}] render_overlay failed: {e}")
        return jsonify({"error": "Overlay rendering failed"}), 500

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.dcm']:
            return jsonify({"error": "Only .jpg and .dcm formats supported"}), 400

        unique_id = uuid.uuid4().hex
        input_path = os.path.join(INFILE, f"{unique_id}{ext}")
        file.save(input_path)

        try:
            processor = XRayProcessor(input_path, OUTFILE, unique_id)
            processed_id, result_data = processor.process()

            do_path = os.path.join(OUTFILE, f"DO-{unique_id}.safetensors")
            #ao_path = os.path.join(OUTFILE, f"AO-{unique_id}.safetensors")
            oo_path = os.path.join(OUTFILE, f"OO-{unique_id}.safetensors")
            missing_files = [p for p in (do_path, oo_path) if not os.path.isfile(p)]
            if missing_files:
                return jsonify({"error": f"Missing files: {', '.join(os.path.basename(f) for f in missing_files)}"}), 500

            result_data["id"] = unique_id
            result_data["input_img"] = file.filename
            view_url = url_for('view', image_id=unique_id, _external=True)
            return jsonify({"success": True, "view_url": view_url, "result": result_data})

        except Exception as e:
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    return render_template('file_uploaderv3.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'prod'], default='dev', help='Run mode: dev or prod')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--threads', type=int, default=8, help='Threads for prod mode')
    args = parser.parse_args()

    if args.mode == 'prod':
        from waitress import serve
        print(f"Starting production server on {args.host}:{args.port} with {args.threads} threads")
        serve(app, host=args.host, port=args.port, threads=args.threads)
    else:
        print(f"Starting development server on {args.host}:{args.port} (threaded)")
        app.run(host=args.host, port=args.port, debug=True, threaded=True)
