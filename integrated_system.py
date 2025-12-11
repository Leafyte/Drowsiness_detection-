"""
Complete Integrated Drowsiness Detection System
Combines MediaPipe detection, CNN classification, and Flask backend
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from collections import deque
import time
from flask import Flask, render_template_string, Response, jsonify
from flask_cors import CORS
import threading
from datetime import datetime
import json

# ==================== CORE DETECTOR ====================
class IntegratedDrowsinessDetector:
    def __init__(self, use_cnn=False, cnn_model_path=None):
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 291, 0, 17, 269, 405]
        
        # Thresholds
        self.EAR_THRESH = 0.21
        self.MAR_THRESH = 0.6
        self.CONSEC_FRAMES = 20
        
        # Counters
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.blink_detected = False
        self.previous_ear = 0.3
        
        # History
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        
        # CNN Model (optional)
        self.use_cnn = use_cnn
        self.cnn_model = None
        if use_cnn and cnn_model_path:
            try:
                from tensorflow import keras
                self.cnn_model = keras.models.load_model(cnn_model_path)
                print(f"CNN model loaded from {cnn_model_path}")
            except Exception as e:
                print(f"Could not load CNN model: {e}")
                self.use_cnn = False
        
        # Status
        self.drowsy = False
        self.yawning = False
        self.session_start = time.time()
        self.drowsy_events = []
    
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[1], mouth[5])
        B = distance.euclidean(mouth[2], mouth[4])
        C = distance.euclidean(mouth[0], mouth[3])
        return (A + B) / (2.0 * C)
    
    def detect_blink(self, ear):
        blink = False
        if self.previous_ear > self.EAR_THRESH and ear < self.EAR_THRESH:
            self.blink_detected = True
        elif self.blink_detected and ear > self.EAR_THRESH:
            blink = True
            self.total_blinks += 1
            self.blink_detected = False
        self.previous_ear = ear
        return blink
    
    def extract_eye_crop(self, frame, eye_landmarks):
        x_coords = [lm[0] for lm in eye_landmarks]
        y_coords = [lm[1] for lm in eye_landmarks]
        x_min = max(0, int(min(x_coords)) - 10)
        y_min = max(0, int(min(y_coords)) - 10)
        x_max = min(frame.shape[1], int(max(x_coords)) + 10)
        y_max = min(frame.shape[0], int(max(y_coords)) + 10)
        
        eye_crop = frame[y_min:y_max, x_min:x_max]
        if eye_crop.size > 0:
            return cv2.resize(eye_crop, (64, 64))
        return None
    
    def predict_cnn(self, eye_crop):
        if self.cnn_model is None or eye_crop is None:
            return None
        try:
            gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
            normalized = gray / 255.0
            input_img = normalized.reshape(1, 64, 64, 1)
            prediction = self.cnn_model.predict(input_img, verbose=0)[0][0]
            return prediction
        except:
            return None
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        data = {
            'ear': 0.0, 'mar': 0.0, 'drowsy': False, 'yawning': False,
            'blink_detected': False, 'total_blinks': self.total_blinks,
            'face_detected': False, 'cnn_left': None, 'cnn_right': None
        }
        
        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame, data
        
        data['face_detected'] = True
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        get_landmarks = lambda indices: [(int(face_landmarks.landmark[i].x * w),
                                         int(face_landmarks.landmark[i].y * h))
                                        for i in indices]
        
        left_eye = get_landmarks(self.LEFT_EYE)
        right_eye = get_landmarks(self.RIGHT_EYE)
        mouth = get_landmarks(self.MOUTH)
        
        # Calculate metrics
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        ear_smooth = np.mean(self.ear_history)
        
        mar = self.mouth_aspect_ratio(mouth)
        self.mar_history.append(mar)
        mar_smooth = np.mean(self.mar_history)
        
        data['ear'] = ear_smooth
        data['mar'] = mar_smooth
        data['blink_detected'] = self.detect_blink(ear_smooth)
        
        # CNN predictions
        if self.use_cnn:
            left_crop = self.extract_eye_crop(frame, left_eye)
            right_crop = self.extract_eye_crop(frame, right_eye)
            data['cnn_left'] = self.predict_cnn(left_crop)
            data['cnn_right'] = self.predict_cnn(right_crop)
        
        # Visualize
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 255), 1)
        
        # Drowsiness detection
        if ear_smooth < self.EAR_THRESH:
            self.eye_closed_counter += 1
            if self.eye_closed_counter >= self.CONSEC_FRAMES:
                self.drowsy = True
                data['drowsy'] = True
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if not self.drowsy_events or time.time() - self.drowsy_events[-1] > 5:
                    self.drowsy_events.append(time.time())
        else:
            self.eye_closed_counter = 0
            self.drowsy = False
        
        # Yawn detection
        if mar_smooth > self.MAR_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= 10:
                self.yawning = True
                data['yawning'] = True
                cv2.putText(frame, "Yawning Detected", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            self.yawn_counter = 0
            self.yawning = False
        
        # Display info
        cv2.putText(frame, f"EAR: {ear_smooth:.2f}", (w - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar_smooth:.2f}", (w - 180, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (w - 180, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Closed: {self.eye_closed_counter}/{self.CONSEC_FRAMES}",
                   (w - 180, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, data

# ==================== FLASK APPLICATION ====================
app = Flask(__name__)
CORS(app)

class VideoMonitor:
    def __init__(self):
        self.detector = IntegratedDrowsinessDetector(use_cnn=False)
        self.camera = None
        self.lock = threading.Lock()
        self.detection_history = deque(maxlen=100)
        self.alert_log = []
        self.current_status = {
            'drowsy': False, 'yawning': False, 'ear': 0.0, 'mar': 0.0,
            'blinks': 0, 'face_detected': False, 'timestamp': datetime.now().isoformat()
        }
        self.running = False
    
    def start(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.running = True
            return True
        return False
    
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def generate_frames(self):
        while self.running and self.camera:
            success, frame = self.camera.read()
            if not success:
                break
            
            processed, data = self.detector.process_frame(frame)
            
            with self.lock:
                self.current_status = {
                    'drowsy': data['drowsy'], 'yawning': data['yawning'],
                    'ear': data['ear'], 'mar': data['mar'],
                    'blinks': data['total_blinks'],
                    'face_detected': data['face_detected'],
                    'timestamp': datetime.now().isoformat()
                }
                
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'ear': data['ear'], 'mar': data['mar'], 'drowsy': data['drowsy']
                })
                
                if data['drowsy']:
                    self.alert_log.append({
                        'type': 'drowsiness',
                        'timestamp': datetime.now().isoformat(),
                        'ear': data['ear']
                    })
                
                if data['yawning']:
                    self.alert_log.append({
                        'type': 'yawn',
                        'timestamp': datetime.now().isoformat(),
                        'mar': data['mar']
                    })
            
            ret, buffer = cv2.imencode('.jpg', processed)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

monitor = VideoMonitor()

# Read the dashboard HTML (you need to save the HTML file as 'templates/dashboard.html')
# Or use render_template_string with the HTML content
DASHBOARD_HTML = """
<!-- Paste the dashboard HTML here or use render_template('dashboard.html') -->
"""

@app.route('/')
def index():
    # Use render_template('dashboard.html') after saving the HTML
    return "Dashboard HTML goes here - save dashboard.html in templates folder"

@app.route('/video_feed')
def video_feed():
    return Response(monitor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    with monitor.lock:
        return jsonify(monitor.current_status)

@app.route('/api/history')
def get_history():
    with monitor.lock:
        return jsonify({'history': list(monitor.detection_history)})

@app.route('/api/alerts')
def get_alerts():
    with monitor.lock:
        return jsonify({
            'alerts': monitor.alert_log[-50:],
            'total_alerts': len(monitor.alert_log)
        })

@app.route('/api/stats')
def get_stats():
    with monitor.lock:
        if monitor.detection_history:
            avg_ear = np.mean([d['ear'] for d in monitor.detection_history])
            avg_mar = np.mean([d['mar'] for d in monitor.detection_history])
        else:
            avg_ear = avg_mar = 0.0
        
        return jsonify({
            'start_time': datetime.now().isoformat(),
            'total_alerts': len(monitor.alert_log),
            'avg_ear': float(avg_ear),
            'avg_mar': float(avg_mar)
        })

@app.route('/api/start', methods=['POST'])
def start_detection():
    success = monitor.start()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    monitor.stop()
    return jsonify({'success': True})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    with monitor.lock:
        monitor.detection_history.clear()
        monitor.alert_log.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    print("=" * 60)
    print("DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5008")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5008, threaded=True)