import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from collections import deque
import time

class DrowsinessDetector:
    def __init__(self):
        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmarks indices for MediaPipe (468 landmarks)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks for yawn detection
        self.MOUTH = [61, 291, 0, 17, 269, 405]
        
        # Thresholds
        self.EAR_THRESH = 0.21
        self.MAR_THRESH = 0.6
        self.CONSEC_FRAMES = 20
        
        # Counters
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        
        # Blink detection
        self.blink_detected = False
        self.previous_ear = 0.3
        
        # History for smoothing
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        
        # Status tracking
        self.drowsy = False
        self.yawning = False
        
        # Session stats
        self.session_start = time.time()
        self.drowsy_events = []
        
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Vertical distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio for yawn detection"""
        # Vertical distances
        A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[5])
        B = distance.euclidean(mouth_landmarks[2], mouth_landmarks[4])
        # Horizontal distance
        C = distance.euclidean(mouth_landmarks[0], mouth_landmarks[3])
        
        mar = (A + B) / (2.0 * C)
        return mar
    
    def extract_eye_region(self, frame, eye_landmarks):
        """Extract eye region for CNN model input"""
        # Get bounding box for eye
        x_coords = [lm[0] for lm in eye_landmarks]
        y_coords = [lm[1] for lm in eye_landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        eye_crop = frame[y_min:y_max, x_min:x_max]
        
        # Resize for CNN input (e.g., 64x64)
        if eye_crop.size > 0:
            eye_crop = cv2.resize(eye_crop, (64, 64))
            return eye_crop, (x_min, y_min, x_max, y_max)
        return None, None
    
    def detect_blink(self, current_ear):
        """Detect blinks based on EAR changes"""
        blink = False
        
        # Blink: EAR drops below threshold then rises back
        if self.previous_ear > self.EAR_THRESH and current_ear < self.EAR_THRESH:
            self.blink_detected = True
        elif self.blink_detected and current_ear > self.EAR_THRESH:
            blink = True
            self.total_blinks += 1
            self.blink_detected = False
        
        self.previous_ear = current_ear
        return blink
    
    def process_frame(self, frame):
        """Main processing function for each frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        detection_data = {
            'ear': 0.0,
            'mar': 0.0,
            'drowsy': False,
            'yawning': False,
            'blink_detected': False,
            'total_blinks': self.total_blinks,
            'left_eye_crop': None,
            'right_eye_crop': None,
            'face_detected': False
        }
        
        if not results.multi_face_landmarks:
            return frame, detection_data
        
        detection_data['face_detected'] = True
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract landmark coordinates
        def get_landmarks(indices):
            return [(int(face_landmarks.landmark[idx].x * w),
                    int(face_landmarks.landmark[idx].y * h))
                    for idx in indices]
        
        left_eye = get_landmarks(self.LEFT_EYE)
        right_eye = get_landmarks(self.RIGHT_EYE)
        mouth = get_landmarks(self.MOUTH)
        
        # Calculate EAR
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        ear_smooth = np.mean(self.ear_history)
        
        # Calculate MAR
        mar = self.mouth_aspect_ratio(mouth)
        self.mar_history.append(mar)
        mar_smooth = np.mean(self.mar_history)
        
        detection_data['ear'] = ear_smooth
        detection_data['mar'] = mar_smooth
        
        # Blink detection
        if self.detect_blink(ear_smooth):
            detection_data['blink_detected'] = True
        
        # Extract eye crops for CNN
        left_crop, left_bbox = self.extract_eye_region(frame, left_eye)
        right_crop, right_bbox = self.extract_eye_region(frame, right_eye)
        detection_data['left_eye_crop'] = left_crop
        detection_data['right_eye_crop'] = right_crop
        
        # Draw eye and mouth contours
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 255), 1)
        
        # Drowsiness detection logic
        if ear_smooth < self.EAR_THRESH:
            self.eye_closed_counter += 1
            if self.eye_closed_counter >= self.CONSEC_FRAMES:
                self.drowsy = True
                detection_data['drowsy'] = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Log event
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
                detection_data['yawning'] = True
                cv2.putText(frame, "YAWNING DETECTED!", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            self.yawn_counter = 0
            self.yawning = False
        
        # Display metrics
        cv2.putText(frame, f"EAR: {ear_smooth:.2f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar_smooth:.2f}", (w - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (w - 150, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Closed: {self.eye_closed_counter}/{self.CONSEC_FRAMES}", 
                   (w - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, detection_data
    
    def get_session_stats(self):
        """Get statistics for the current session"""
        duration = time.time() - self.session_start
        return {
            'session_duration': duration,
            'total_blinks': self.total_blinks,
            'drowsy_events': len(self.drowsy_events),
            'blink_rate': self.total_blinks / (duration / 60) if duration > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting drowsiness detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, detection_data = detector.process_frame(frame)
        
        cv2.imshow("Drowsiness Detection", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print session stats
    stats = detector.get_session_stats()
    print("\n=== Session Statistics ===")
    print(f"Duration: {stats['session_duration']:.1f} seconds")
    print(f"Total Blinks: {stats['total_blinks']}")
    print(f"Drowsy Events: {stats['drowsy_events']}")
    print(f"Blink Rate: {stats['blink_rate']:.1f} blinks/min")
    
    cap.release()
    cv2.destroyAllWindows()