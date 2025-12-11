from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import json
import threading
from datetime import datetime
from collections import deque
import numpy as np

# Import the detector (assuming it's in drowsiness_detector.py)
# from drowsiness_detector import DrowsinessDetector

app = Flask(__name__)
CORS(app)

class DrowsinessMonitor:
    def __init__(self):
        # Uncomment when you have the detector module
        # self.detector = DrowsinessDetector()
        self.camera = None
        self.output_frame = None
        self.lock = threading.Lock()
        
        # Real-time data storage
        self.detection_history = deque(maxlen=100)
        self.alert_log = []
        self.current_status = {
            'drowsy': False,
            'yawning': False,
            'ear': 0.0,
            'mar': 0.0,
            'blinks': 0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Statistics
        self.session_stats = {
            'start_time': datetime.now().isoformat(),
            'total_alerts': 0,
            'total_blinks': 0,
            'avg_ear': 0.0,
            'avg_mar': 0.0
        }
        
    def start_camera(self):
        """Initialize camera feed"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            return True
        return False
    
    def stop_camera(self):
        """Release camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def generate_frames(self):
        """Generator function for video streaming"""
        self.start_camera()
        
        while True:
            if self.camera is None:
                break
                
            success, frame = self.camera.read()
            if not success:
                break
            
            # Process frame with detector
            # Uncomment when detector is ready
            # processed_frame, detection_data = self.detector.process_frame(frame)
            
            # Temporary: Just use original frame
            processed_frame = frame
            detection_data = {
                'ear': np.random.uniform(0.2, 0.3),
                'mar': np.random.uniform(0.3, 0.5),
                'drowsy': False,
                'yawning': False,
                'blink_detected': False,
                'total_blinks': 0,
                'face_detected': True
            }
            
            # Update status
            with self.lock:
                self.current_status = {
                    'drowsy': detection_data['drowsy'],
                    'yawning': detection_data['yawning'],
                    'ear': detection_data['ear'],
                    'mar': detection_data['mar'],
                    'blinks': detection_data['total_blinks'],
                    'face_detected': detection_data['face_detected'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store history
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'ear': detection_data['ear'],
                    'mar': detection_data['mar'],
                    'drowsy': detection_data['drowsy']
                })
                
                # Log alerts
                if detection_data['drowsy']:
                    self.alert_log.append({
                        'type': 'drowsiness',
                        'timestamp': datetime.now().isoformat(),
                        'ear': detection_data['ear']
                    })
                    self.session_stats['total_alerts'] += 1
                
                if detection_data['yawning']:
                    self.alert_log.append({
                        'type': 'yawn',
                        'timestamp': datetime.now().isoformat(),
                        'mar': detection_data['mar']
                    })
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Initialize monitor
monitor = DrowsinessMonitor()

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(monitor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    with monitor.lock:
        return jsonify(monitor.current_status)

@app.route('/api/history')
def get_history():
    """Get detection history for charts"""
    with monitor.lock:
        return jsonify({
            'history': list(monitor.detection_history),
            'total_points': len(monitor.detection_history)
        })

@app.route('/api/alerts')
def get_alerts():
    """Get alert log"""
    with monitor.lock:
        # Return last 50 alerts
        return jsonify({
            'alerts': monitor.alert_log[-50:],
            'total_alerts': len(monitor.alert_log)
        })

@app.route('/api/stats')
def get_stats():
    """Get session statistics"""
    with monitor.lock:
        # Calculate averages
        if monitor.detection_history:
            avg_ear = np.mean([d['ear'] for d in monitor.detection_history])
            avg_mar = np.mean([d['mar'] for d in monitor.detection_history])
        else:
            avg_ear = 0.0
            avg_mar = 0.0
        
        stats = monitor.session_stats.copy()
        stats['avg_ear'] = float(avg_ear)
        stats['avg_mar'] = float(avg_mar)
        stats['current_time'] = datetime.now().isoformat()
        
        return jsonify(stats)

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start camera and detection"""
    success = monitor.start_camera()
    return jsonify({'success': success, 'message': 'Detection started' if success else 'Already running'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop camera and detection"""
    monitor.stop_camera()
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session data"""
    with monitor.lock:
        monitor.detection_history.clear()
        monitor.alert_log.clear()
        monitor.session_stats = {
            'start_time': datetime.now().isoformat(),
            'total_alerts': 0,
            'total_blinks': 0,
            'avg_ear': 0.0,
            'avg_mar': 0.0
        }
    return jsonify({'success': True, 'message': 'Session reset'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008, threaded=True)