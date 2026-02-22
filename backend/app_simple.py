from flask import Flask, send_from_directory, Response, request, jsonify

import cv2
import json
import os
import base64
import io  # <--- NEW: Handles images in RAM (Privacy Protection)
import numpy as np
import mediapipe as mp
import replicate
from datetime import datetime
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# --- Load Environment Variables ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=env_path)

# Debug: Print API Token status
api_token = os.getenv("REPLICATE_API_TOKEN")
if api_token:
    print(f"✅ Replicate Token Loaded: {api_token[:4]}...{api_token[-4:]}")
else:
    print("❌ ERROR: Replicate Token NOT found. Check your .env file!")

# Debug: Print Decart Token status
decart_key = os.getenv("DECART_API_KEY")
if decart_key:
    print(f"✅ Decart Key Loaded: {decart_key[:4]}...{decart_key[-4:]}")
else:
    print("❌ ERROR: Decart Key NOT found. Check your .env file!")

# --- Flask setup ---
app = Flask(__name__, static_folder="../ui", template_folder="../ui")


# --- Load trained skin tone model ---
try:
    model_path = os.path.join(os.path.dirname(__file__), "skin_tone_cnn.h5")
    skin_model = load_model(model_path)
    skin_classes = ["Chinito/Chinita", "Mestizo/Mestiza", "Moreno/Morena"]
    print(f"✅ Skin Model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Skin model failed. Error: {e}")
    skin_model = None
    skin_classes = ["Unknown"]

# --- MediaPipe Face Detection ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- Camera setup ---
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ==========================================
#       DATA LOADING
# ==========================================

# 1. LOAD WARDROBE DATABASE
def load_wardrobe():
    try:
        json_path = os.path.join(os.path.dirname(__file__), "wardrobe.json")
        with open(json_path, 'r') as f:
            print("✅ Wardrobe database loaded!")
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading wardrobe.json: {e}")
        return {}

WARDROBE = load_wardrobe()

# 2. DEFINE SKIN TONE RULES
SKIN_TONE_RULES = {
    "Moreno/Morena": ["black", "white", "blue", "green", "red", "brown"],
    "Mestizo/Mestiza": ["black", "white", "gray", "beige", "blue", "green", "red", "violet", "skyblue"],
    "Chinito/Chinita": ["black", "white", "blue", "gray", "red", "green", "brown"],
    "Dark": ["white", "yellow", "red", "orange", "green", "blue", "beige"]
}

# ==========================================
#               CORE ROUTES
# ==========================================

@app.route('/')
def home():
    return send_from_directory(app.template_folder, "index.html")

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
#           HELPER FUNCTIONS
# ==========================================

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Simple Face Box Drawing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def predict_skin_tone(frame_count=5):
    if skin_model is None: return "Unknown"
    predictions = []
    print("--- Starting Skin Analysis ---")
    for i in range(frame_count):
        success, frame = camera.read()
        if not success: continue
        
        # Crop Face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
            x2, y2 = min(x1 + int(bbox.width * w), w), min(y1 + int(bbox.height * h), h)
            face_img = frame[y1:y2, x1:x2]
        else:
            face_img = frame # Fallback to full frame

        if face_img.size == 0: continue
        img = cv2.resize(face_img, (180, 180)) 
        img = np.expand_dims(img, axis=0)
        pred = skin_model.predict(img)
        detected_class = skin_classes[np.argmax(pred)]
        predictions.append(detected_class)
        print(f"📸 Frame {i+1}: {detected_class}")

    if predictions:
        final = max(set(predictions), key=predictions.count)
        print(f"🎨 FINAL RESULT: {final}")
        return final
    return "Unknown"

# ==========================================
#           API ENDPOINTS
# ==========================================

@app.route('/start-camera', methods=['POST'])
def start_camera():
    try:
        data = request.get_json()
        event = data.get("event", "unknown")
        gender = data.get("gender", "unknown")
        
        # Detect Skin Tone
        skin_tone = predict_skin_tone(frame_count=5)

        session_data = {
            "status": "active", "event": event, "gender": gender,
            "skin_tone": skin_tone,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to data.json
        data_path = os.path.join(os.path.dirname(__file__), "data.json")
        with open(data_path, "w") as f:
            json.dump(session_data, f, indent=2)

        return jsonify({"message": "Camera session started", "data": session_data})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend_outfit():
    try:
        data = request.get_json()
        skin_tone = data.get("skin_tone", "Moreno/Morena")
        event = data.get("event", "casual").lower()
        gender = data.get("gender", "male").lower()

        print(f"🔎 Looking for outfits: Gender={gender}, Skin={skin_tone}, Event={event}")

        allowed_colors = SKIN_TONE_RULES.get(skin_tone, ["black", "white"])
        recommendations = []

        if gender in WARDROBE and event in WARDROBE[gender]:
            category_data = WARDROBE[gender][event]
            
            for color_key, items in category_data.items():
                if color_key.lower() in allowed_colors:
                    for item in items:
                        item['color_category'] = color_key
                        recommendations.append(item)
        else:
            print(f"   ❌ No outfits found for {gender} + {event}")

        print(f"👕 Sending {len(recommendations)} suggestions")
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"❌ Error in recommend: {e}")
        return jsonify({"error": str(e)}), 500

# --- NEW: DECART VIRTUAL TRY-ON ROUTE ---
@app.route('/api/decart-token', methods=['POST'])
def handle_decart_token():
    """
    Directly fetches the DECART_API_KEY from the .env file 
    and securely sends it to the frontend for the Live Mirror.
    """
    api_key = os.getenv("DECART_API_KEY")
    
    if api_key:
        return jsonify({"apiKey": api_key})
    else:
        print("❌ ERROR: DECART_API_KEY is missing from your .env file!")
        return jsonify({"error": "No API key found in .env"}), 500


# --- PRIVACY-FOCUSED TRY-ON (RAM ONLY) ---
@app.route('/generate-tryon', methods=['POST'])
def generate_tryon():
    try:
        print("🚀 Starting AI Try-On Generation (Privacy Mode)...")
        data = request.get_json()
        user_image_data = data.get("user_image") 
        shirt_path_rel = data.get("shirt_path") 

        if not user_image_data or not shirt_path_rel:
            return jsonify({"error": "Missing data"}), 400

        # 1. Clean Path and Verify Shirt Exists
        if shirt_path_rel.startswith("/"): shirt_path_rel = shirt_path_rel[1:]
        shirt_full_path = os.path.join(app.static_folder, shirt_path_rel)
        
        if not os.path.exists(shirt_full_path):
            print(f"❌ ERROR: Shirt file missing at {shirt_full_path}")
            return jsonify({"error": f"Shirt file not found: {shirt_path_rel}"}), 404

        # 2. Process User Image in MEMORY (No File Saved)
        if "base64," in user_image_data:
            user_image_data = user_image_data.split("base64,")[1]
        
        # Decode base64 to bytes
        user_img_bytes = base64.b64decode(user_image_data)
        
        # Create a RAM-based file object
        user_file_object = io.BytesIO(user_img_bytes)

        # 3. Call Replicate
        # We open the shirt file (local), but pass the user file from RAM
        with open(shirt_full_path, "rb") as shirt_file:
            print(f"⏳ Sending request to Replicate...")
            output = replicate.run(
                "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
                input={
                    "garm_img": shirt_file,        # Local file
                    "human_img": user_file_object, # RAM object
                    "garment_des": "clothing item",
                    "crop": False,
                    "seed": 42
                }
            )

        print(f"✅ Success: {output}")
        return jsonify({"generated_image": str(output)})

    except Exception as e:
        print(f"❌ Error in generate_tryon: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/data')
def get_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                return jsonify(json.load(f))
        return jsonify({"status": "inactive"})
    except Exception:
        return jsonify({"error": "Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)