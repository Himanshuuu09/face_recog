import cv2
import face_recognition
import numpy as np
from fer import FER
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import mediapipe as mp
from flask_cors import CORS
import threading

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000"])
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000"]}})
emotion_detector = FER()

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

reference_frame = None
reference_encoding = None
frame_counter = 0  # For controlling the frequency of checks
background_frame_interval = 20  # Check background every 20 frames
face_detection_interval = 5  # Perform face detection every 5 frames
emotion_detection_interval = 10  # Perform emotion detection every 10 frames
background_reference = None  # For background change detection reference

def get_face_encoding(frame, detection):
    h, w, _ = frame.shape
    box = detection.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)

    face_roi = frame[y:y + height, x:x + width]
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(face_rgb)
    if len(face_encodings) == 0:
        raise ValueError("No face encoding found in the video frame.")
    
    return face_encodings[0]

def compare_faces(reference_encoding, video_frame, detection):
    h, w, _ = video_frame.shape
    box = detection.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)

    face_roi = video_frame[y:y + height, x:x + width]
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(face_rgb)
    if len(face_encodings) == 0:
        raise ValueError("No face encoding found in the video frame.")
    
    match = face_recognition.compare_faces([reference_encoding], face_encodings[0], tolerance=0.6)
    return match[0]

def detect_emotion(video_frame):
    emotion_data = emotion_detector.detect_emotions(video_frame)
    if not emotion_data:
        emit('message', {'text': "No face detected for emotion analysis."})
        return
    
    top_emotion = emotion_detector.top_emotion(video_frame)
    emotion, score = top_emotion
    emit('message', {'text': f"Top emotion detected: {emotion} (score: {score:.2f})"})
    
    suspicious_emotions = ['angry', 'disgust', 'fear', 'sad']
    if emotion in suspicious_emotions:
        emit('message', {'text': "Suspicious emotion detected. Possible cheating!"})
    else:
        emit('message', {'text': "No suspicious emotion detected."})

def check_background_change(video_frame, reference_frame, threshold=23):
    diff = cv2.absdiff(video_frame, reference_frame)
    mean_diff = np.mean(diff)
    emit('message', {'text': f"Background change mean difference: {mean_diff:.2f} (threshold: {threshold:.2f})"})
    return mean_diff > threshold 

@socketio.on('video_data')
def handle_video_frame(data):
    global reference_frame, reference_encoding, frame_counter, background_reference

    np_data = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        emit('message', {'text': 'Invalid or empty frame received.'})
        return

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Increment frame counter
    frame_counter += 1

    # Only perform face detection every `face_detection_interval` frames
    if frame_counter % face_detection_interval == 0:
        results = face_detection.process(rgb_frame)
        
        # Ensure results are not None and contain detections
        if results is None or results.detections is None or len(results.detections) == 0:
            emit('message', {'text': 'Error processing frame or no faces detected.'})
            return

        # Set reference encoding if it's the first frame
        if reference_frame is None:
            if len(results.detections) > 1:
                emit('message', {'text': "Multiple faces detected in the reference frame."})
                return

            detection = results.detections[0]
            try:
                reference_encoding = get_face_encoding(frame, detection)
            except ValueError as e:
                emit('message', {'text': f"Error: {e}"})
                return
            reference_frame = frame
            background_reference = frame  # Set the initial reference for background change detection
            emit('message', {'text': "Reference frame set."})
            return

        # Check face match if one face is detected
        if len(results.detections) == 1:
            face_match = compare_faces(reference_encoding, frame, results.detections[0])
            if not face_match:
                emit('message', {'text': "Face mismatch detected. Possible cheating!"})

    # Only check for background change every `background_frame_interval` frames
    if background_reference is not None and frame_counter % background_frame_interval == 0:
        if check_background_change(frame, background_reference):
            emit('message', {'text': "Significant background change detected. Possible cheating!"})

    # Only perform emotion detection every `emotion_detection_interval` frames
    if frame_counter % emotion_detection_interval == 0:
        # Run emotion detection in a separate thread
        threading.Thread(target=detect_emotion, args=(frame,)).start()

    # Ensure results and detections exist before using them
    if 'results' in locals() and results.detections is not None and len(results.detections) == 1:
        detection = results.detections[0]
        box = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        width = int(box.width * w)
        height = int(box.height * h)

        face_data = {"x": x, "y": y, "width": width, "height": height}
        emit('result', {"face": face_data})
    else:
        emit('result', {"face": None})



@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
