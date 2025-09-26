# --- app_simple.py ---
from flask import Flask, send_from_directory, Response, request, jsonify
import cv2
import json
import os
import random
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# --- Flask setup ---
app = Flask(__name__, static_folder="../ui", template_folder="../ui")

# --- Load trained skin tone model ---
model_path = os.path.join(os.path.dirname(__file__), "skin_tone_cnn.h5")
skin_model = load_model(model_path)
skin_classes = ["dark", "mid-dark", "mid-light", "light"]

# --- MediaPipe Face Detection ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- Camera setup (DirectShow for Windows) ---
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise Exception("Cannot open webcam. Close other apps using the camera.")

# --- Routes for UI ---
@app.route('/')
def home():
    return send_from_directory(app.template_folder, "index.html")

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# --- Camera Stream with live face box ---
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Body type classification (temporary) ---
def classify_body_type(frame, gender):
    body_types_male = ["Rectangle", "Triangle", "Inverted Triangle", "Oval"]
    body_types_female = ["Hourglass", "Pear", "Apple", "Rectangle"]
    if gender.lower() == "male":
        return random.choice(body_types_male)
    elif gender.lower() == "female":
        return random.choice(body_types_female)
    else:
        return "Unknown"

# --- Skin tone prediction with frame averaging ---
def predict_skin_tone(frame_count=5):
    predictions = []
    for i in range(frame_count):
        success, frame = camera.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(x1 + int(bbox.width * w), w)
            y2 = min(y1 + int(bbox.height * h), h)
            face_img = frame[y1:y2, x1:x2]
        else:
            face_img = frame

        # Normalize brightness
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v_)
        hsv_eq = cv2.merge([h_, s_, v_eq])
        frame_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

        img = cv2.resize(frame_eq, (64, 64)) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = skin_model.predict(img)
        class_index = np.argmax(pred)
        predictions.append(skin_classes[class_index])
        print(f"Frame {i+1} prediction: {skin_classes[class_index]} | Probabilities: {pred[0]}")

    if predictions:
        skin_tone = max(set(predictions), key=predictions.count)
        print(f"🎨 Final skin tone: {skin_tone}")
        return skin_tone
    else:
        return "unknown"

# --- API: Start Camera Session ---
@app.route('/start-camera', methods=['POST'])
def start_camera():
    try:
        data = request.get_json()
        event = data.get("event", "unknown")
        gender = data.get("gender", "unknown")

        success, frame = camera.read()
        if not success:
            body_type = "Unknown"
        else:
            body_type = classify_body_type(frame, gender)

        skin_tone = predict_skin_tone(frame_count=5)

        session_data = {
            "status": "active",
            "event": event,
            "gender": gender,
            "skin_tone": skin_tone,
            "body_type": body_type,
            "timestamp": datetime.now().isoformat()
        }

        data_path = os.path.join(os.path.dirname(__file__), "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                old_data = json.load(f)
        else:
            old_data = {}

        old_data.update(session_data)
        with open(data_path, "w") as f:
            json.dump(old_data, f, indent=2)

        print("✅ Session saved:", session_data)
        return jsonify({"message": "Camera session started", "data": old_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- API: Get Analysis Data ---
@app.route('/data')
def get_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            data = {"status": "inactive"}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
