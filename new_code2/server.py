import cv2
import face_recognition
import numpy as np
from fer import FER
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import mediapipe as mp
from flask_cors import CORS

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000"])
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000"]}})
emotion_detector = FER()

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

reference_frame = None
reference_encoding = None
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

@socketio.on('video_data')
def handle_video_frame(data):
    global reference_frame, reference_encoding

    np_data = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        emit('message', {'text': 'Invalid or empty frame received.'})
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results is None or results.detections is None:
        emit('message', {'text': 'Error processing frame or no faces detected.'})
        return

    if reference_frame is None:
        if len(results.detections) == 0:
            emit('message', {'text': "No face detected in the reference frame."})
            return
        elif len(results.detections) > 1:
            emit('message', {'text': "Multiple faces detected in the reference frame."})
            return

        detection = results.detections[0]
        try:
            reference_encoding = get_face_encoding(frame, detection)
        except ValueError as e:
            emit('message', {'text': f"Error: {e}"})
            return
        reference_frame = frame
        emit('message', {'text': "Reference frame set."})
        return

    try:
        if len(results.detections) == 1:
            face_match = compare_faces(reference_encoding, frame, results.detections[0])
            if not face_match:
                emit('message', {'text': "Face mismatch detected. Possible cheating!"})

        detect_emotion(frame)

        if check_background_change(frame, reference_frame):
            emit('message', {'text': "Significant background change detected. Possible cheating!"})

        if len(results.detections) == 1:
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

    except ValueError as e:
        emit('message', {'text': f"Error: {e}"})
        emit('result', {"face": None})

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
