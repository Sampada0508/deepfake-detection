# backend/app.py
# ===============================================================
# TRUTHLENS - DEEPVISION AI (Flask Backend) - full app.py
# ===============================================================
import os
import traceback
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

# ML imports (if not available or model files missing we handle gracefully)
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models
    from PIL import Image
    import cv2
    import numpy as np
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional OpenAI support (disabled by default)
USE_OPENAI = False
try:
    if os.environ.get('OPENAI_API_KEY'):
        import openai
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# ===============================================================
# Flask app setup
# ===============================================================
app = Flask(__name__)
# Ensure you run python from project root or adjust paths accordingly
app.config['UPLOAD_FOLDER'] = os.path.join('backend', 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================================================
# Global in-memory stores
# ===============================================================
PREDICTIONS = []  # {id, file, type, label, confidence, time}
CHAT_SESSIONS = defaultdict(lambda: {"messages": [], "created": datetime.now().isoformat()})

# ===============================================================
# Model loading (attempt)
# ===============================================================
image_model = None
video_model = None
idx_to_class = {0: "fake", 1: "real"}

if TORCH_AVAILABLE:
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    except Exception:
        device = torch.device("cpu")

    def try_load_image_model(path="models/image/image_model.pth"):
        try:
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 2)
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device).eval()
            app.logger.info(f"Loaded image model from {path}")
            return m
        except Exception as e:
            app.logger.warning(f"Could not load image model from {path}: {e}")
            return None

    def try_load_video_model(path="models/deepfake_resnet18.pth"):
        try:
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 2)
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device).eval()
            app.logger.info(f"Loaded video model from {path}")
            return m
        except Exception as e:
            app.logger.warning(f"Could not load video model from {path}: {e}")
            return None

    image_model = try_load_image_model()
    video_model = try_load_video_model()
else:
    app.logger.warning("Torch not available — prediction endpoints will return errors until dependencies/models are provided.")

# Preprocessing transform
if TORCH_AVAILABLE:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
else:
    transform = None

# ===============================================================
# Helpers
# ===============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_prediction_record(filename, ptype, label, confidence):
    try:
        PREDICTIONS.append({
            "id": len(PREDICTIONS) + 1,
            "file": filename,
            "type": ptype,
            "label": label,
            "confidence": float(confidence),
            "time": now_str()
        })
    except Exception:
        app.logger.error("Failed to append prediction record:\n" + traceback.format_exc())

def logits_to_label_conf(logits):
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return idx_to_class.get(pred_idx, str(pred_idx)), float(probs[pred_idx])

# ===============================================================
# Prediction implementations (image/video)
# ===============================================================
def predict_image_path(image_path):
    if not TORCH_AVAILABLE or image_model is None:
        raise RuntimeError("Image model not loaded. Ensure PyTorch + model file exist.")
    img = Image.open(image_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(next(image_model.parameters()).device)
    with torch.no_grad():
        out = image_model(t)
    label, conf = logits_to_label_conf(out)
    return label, conf

def predict_video_path(video_path, max_samples=30):
    if not TORCH_AVAILABLE or video_model is None:
        raise RuntimeError("Video model not loaded. Ensure PyTorch + model file exist.")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = np.linspace(0, max(0, total_frames - 1), min(max_samples, max(1, total_frames))).astype(int)

    probs_accum = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        t = transform(pil).unsqueeze(0).to(next(video_model.parameters()).device)
        with torch.no_grad():
            out = video_model(t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            probs_accum.append(probs)
    cap.release()

    if not probs_accum:
        raise RuntimeError("No frames read from video or video corrupted.")
    avg_probs = np.mean(probs_accum, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    label = idx_to_class.get(pred_idx, str(pred_idx))
    conf = float(avg_probs[pred_idx])
    return label, conf

# ===============================================================
# Page routes
# ===============================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/articles')
def articles():
    return render_template('articles.html')

@app.route('/article-detail')
def article_detail():
    return render_template('article_detail.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/chat')
def chat_page():
    return render_template('chatbot.html')

# ===============================================================
# API: visualization data
# ===============================================================
@app.route('/api/visualization-data')
def viz_data():
    try:
        today = datetime.now().date()
        days = [(today - timedelta(days=i)) for i in range(13, -1, -1)]
        dates_str = [d.strftime('%Y-%m-%d') for d in days]

        fake_counts = {d: 0 for d in dates_str}
        real_counts = {d: 0 for d in dates_str}

        for rec in PREDICTIONS:
            try:
                rec_time = datetime.strptime(rec['time'], "%Y-%m-%d %H:%M:%S")
                key = rec_time.date().strftime('%Y-%m-%d')
                if key in fake_counts:
                    if rec.get('label') == 'fake':
                        fake_counts[key] += 1
                    elif rec.get('label') == 'real':
                        real_counts[key] += 1
            except Exception:
                continue

        fake_series = [fake_counts[d] for d in dates_str]
        real_series = [real_counts[d] for d in dates_str]

        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for rec in PREDICTIONS:
            try:
                c = float(rec.get('confidence', 0.0))
                if c < 0.2:
                    bins["0.0-0.2"] += 1
                elif c < 0.4:
                    bins["0.2-0.4"] += 1
                elif c < 0.6:
                    bins["0.4-0.6"] += 1
                elif c < 0.8:
                    bins["0.6-0.8"] += 1
                else:
                    bins["0.8-1.0"] += 1
            except Exception:
                continue

        history = list(reversed(PREDICTIONS))[:200]

        data = {
            "dates": dates_str,
            "fake_counts": fake_series,
            "real_counts": real_series,
            "confidence_bins": bins,
            "history": history
        }
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        app.logger.error("viz_data error: " + traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# ===============================================================
# Prediction endpoints
# ===============================================================
@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No image file provided"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        if not TORCH_AVAILABLE or image_model is None:
            raise RuntimeError("Image model not available on server.")
        label, conf = predict_image_path(save_path)
        append_prediction_record(filename, "image", label, conf)
        result = {"label": label, "confidence": conf}
    except Exception as e:
        append_prediction_record(filename, "image", "error", 0.0)
        result = {"error": str(e), "confidence": 0.0}
        app.logger.error("predict_image_route error: " + traceback.format_exc())

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(result)
    return render_template("detect.html", image_result=result)

@app.route('/predict_video', methods=['POST'])
def predict_video_route():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        if not TORCH_AVAILABLE or video_model is None:
            raise RuntimeError("Video model not available on server.")
        label, conf = predict_video_path(save_path)
        append_prediction_record(filename, "video", label, conf)
        result = {"label": label, "confidence": conf}
    except Exception as e:
        append_prediction_record(filename, "video", "error", 0.0)
        result = {"error": str(e), "confidence": 0.0}
        app.logger.error("predict_video_route error: " + traceback.format_exc())

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(result)
    return render_template("detect.html", video_result=result)

# ===============================================================
# API: chat (GET for greeting, POST for messages)
# -------------------------
# Chat API (improved)
# -------------------------
import os
import re
import json
from flask import request

# try optional OpenAI integration
OPENAI_AVAILABLE = False
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
        OPENAI_AVAILABLE = True
        app.logger.info("OpenAI client available for chat fallback.")
    except Exception as e:
        app.logger.warning("OpenAI package not available; continuing with rule-based chat. " + str(e))

BOT_DEFAULT_NAME = "LensBot"

def call_openai_chat(system_prompt, user_message, max_tokens=400):
    """Call OpenAI ChatCompletion (optional). Returns string or None on error."""
    if not OPENAI_AVAILABLE:
        return None
    try:
        # NOTE: using gpt-3.5-turbo as example. Adjust model name as needed.
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error("OpenAI call failed: " + str(e))
        return None

# ---- Utilities for rule matching ----
def contains_any(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)

def match_re(r, text):
    return re.search(r, text, re.I)

# ---- richer rule-based responder ----
def simple_rule_response(message, persona="friendly", bot_name=BOT_DEFAULT_NAME):
    m = message.strip()
    lower = m.lower()

    # Quick facts & greetings
    if contains_any(lower, ["hi", "hello", "hey", "good morning", "good evening", "greetings"]):
        txt = f"Hi — I'm {bot_name}. I can help you with detection, visualizations, and interpreting results. Try: 'How does detection work?' or 'Show visualization'."
        return [{"bot_name": bot_name, "text": txt}]

    if contains_any(lower, ["how does detection work", "how detection work", "detect deepfake", "how do you detect"]):
        txt = (
            "Detection overview: we extract frames, run a CNN (ResNet18) on each frame to score authenticity, "
            "then aggregate frame-level predictions for videos using a majority/averaging strategy. Preprocessing includes resizing, normalization, and augmentations during training."
        )
        return [{"bot_name": bot_name, "text": txt}]

    # Implementation / methods / training
    if contains_any(lower, ["model", "architecture", "resnet", "training", "dataset", "implementation"]):
        txt = (
            "Implementation notes:\n"
            "- Image model: ResNet18 fine-tuned to classify real vs fake frames.\n"
            "- Video pipeline: sample up to N frames per video, classify each, then aggregate probabilities.\n"
            "- Training: cross-entropy loss, standard augmentations (flip, crop), Adam optimizer and LR scheduling.\n"
            "If you want code snippets or the training loop, ask 'show training code'."
        )
        return [{"bot_name": bot_name, "text": txt}]

    # Confidence interpretation
    if contains_any(lower, ["confidence", "how to interpret", "interpret confidence", "what does confidence mean"]):
        txt = (
            "Confidence is the model's predicted probability for the chosen label. "
            "High confidence (e.g. > 0.85) means the model is strongly certain. For borderline values (0.45–0.6) treat results as uncertain and consider manual review."
        )
        return [{"bot_name": bot_name, "text": txt}]

    # Visualization actions
    if contains_any(lower, ["visualization", "show visualization", "show charts", "open visualization", "show me the charts"]):
        txt = "Opening the visualization page for aggregated system analytics."
        return [{"bot_name": bot_name, "text": txt, "action": "show_visualization"}]

    # How to upload / use the web UI
    if contains_any(lower, ["upload", "how to upload", "how to test", "use detect", "detect page"]):
        txt = (
            "To test: go to Detection -> choose Image or Video -> upload the file -> click Detect. "
            "The server will return a label and confidence. Files are stored under `static/uploads` during processing."
        )
        return [{"bot_name": bot_name, "text": txt}]

    # Ask for PDF / article / methods
    if contains_any(lower, ["article", "paper", "methods", "future", "enhancement", "future enhancements"]):
        txt = (
            "Project article: we cover dataset, methods, evaluation, and future improvements such as multimodal features, temporal transformers, better adversarial robustness, and user-facing explainability. "
            "If you want a downloadable PDF of the project report, ask 'download pdf' and I'll prepare it."
        )
        return [{"bot_name": bot_name, "text": txt, "action": "offer_pdf"}]

    # Ask for examples
    if contains_any(lower, ["example", "sample", "show example", "demo"]):
        txt = "Example: Upload a short video (5–10s). The pipeline will sample frames (default 30), run the ResNet, and return the majority label. Try 'How do I run an example?' for step-by-step."
        return [{"bot_name": bot_name, "text": txt}]

    # Direct commands
    if contains_any(lower, ["home", "go home", "open home"]):
        return [{"bot_name": bot_name, "text": "Navigating to home.", "action": "go_home"}]
    if contains_any(lower, ["help", "what can you do", "capabilities"]):
        txt = (
            "I can: 1) Explain detection & confidence, 2) Show visualizations, 3) Help troubleshoot uploads, 4) Prepare a project PDF, 5) Show recent prediction history. "
            "Try commands like: 'Show visualization', 'Explain confidence', 'List recent predictions'."
        )
        return [{"bot_name": bot_name, "text": txt}]

    # Recent predictions / history
    if contains_any(lower, ["recent", "history", "predictions", "show recent", "list recent"]):
        # return a short summary plus an action so the frontend can fetch /api/visualization-data or /api/predictions
        latest = list(reversed(PREDICTIONS))[:6]
        if not latest:
            return [{"bot_name": bot_name, "text": "No predictions found yet — try uploading an image or video."}]
        summary_lines = []
        for r in latest:
            summary_lines.append(f"{r['file']} — {r['label']} ({r['confidence']:.2f}) at {r['time']}")
        txt = "Here are the most recent predictions:\n" + "\n".join(summary_lines)
        return [{"bot_name": bot_name, "text": txt, "action": "show_history"}]

    # Fallback: use OpenAI if available
    if OPENAI_AVAILABLE:
        system = (
            "You are LensBot, assistant for TRUTHLENS — an AI deepfake detection project. "
            "Be concise and helpful; if user asks for navigation use the actions: show_visualization, offer_pdf, show_history, go_home. "
            "If user asks about technical details provide clear, step-by-step answers."
        )
        ai_resp = call_openai_chat(system, m, max_tokens=500)
        if ai_resp:
            return [{"bot_name": bot_name, "text": ai_resp}]

    # Generic fallback answer (expanded)
    txt = (
        "I can help with detection, visualizations, and interpreting results. "
        "Try: 'How does detection work?', 'Show visualization', 'Explain confidence', or 'List recent predictions'. "
        "If you'd like a detailed report, ask 'download pdf' or 'write report'."
    )
    return [{"bot_name": bot_name, "text": txt}]

# ---- chat endpoints ----
@app.route('/api/chat', methods=['GET', 'POST'])
def api_chat():
    """
    GET (optional): returns a welcome/greeting reply
      - query params: session_id, bot_name
    POST (required): accepts JSON {session_id, message, persona, bot_name}
      - returns: {"status":"ok","replies":[{"bot_name":..,"text":.., "action":...}, ...]}
    """
    try:
        if request.method == 'GET':
            bot_name = request.args.get('bot_name', BOT_DEFAULT_NAME)
            # simple welcome
            welcome = f"Hi — I'm {bot_name}. Ask me about detection, viewing visualizations, or interpreting results."
            return jsonify({"status": "ok", "replies":[{"bot_name": bot_name, "text": welcome}]})

        # POST
        body = request.get_json(force=True)
        msg = (body.get('message') or "").strip()
        persona = body.get('persona') or 'friendly'
        bot_name = body.get('bot_name') or BOT_DEFAULT_NAME

        if not msg:
            return jsonify({"status":"error","message":"Empty message"}), 400

        # first try the rule engine (fast)
        replies = simple_rule_response(msg, persona=persona, bot_name=bot_name)

        # replies is a list of dicts {bot_name, text, (optional) action}
        return jsonify({"status":"ok", "replies": replies})
    except Exception as e:
        app.logger.error("api_chat error: " + traceback.format_exc())
        return jsonify({"status":"error","message":str(e)}), 500

# ===============================================================
# Run app
# ===============================================================
# at bottom of backend/app.py — replace existing run block with this:
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)