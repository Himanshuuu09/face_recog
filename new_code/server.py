import cv2
import face_recognition
import numpy as np
from fer import FER  # For emotion detection
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from flask_cors import CORS 

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5500"])  # Allow specific origin
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5500"]}})
emotion_detector = FER()

def preprocess_image(image):
    """Preprocess the image to enhance face detection in dim lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return enhanced

def detect_face(image):
    """Detects face and returns the face encoding if exactly one face is detected."""
    image = preprocess_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 1:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return face_encodings[0]
    else:
        raise ValueError("Face detection failed or multiple faces detected.")

def compare_faces(reference_encoding, video_frame):
    """Compares the face in the reference frame with the face in a video frame."""
    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    frame_encodings = face_recognition.face_encodings(rgb_frame)
    
    if len(frame_encodings) == 0:
        raise ValueError("No face detected in the video frame.")
    elif len(frame_encodings) > 1:
        raise ValueError("More than one person detected in the video. Possible cheating detected.")
    
    return face_recognition.compare_faces([reference_encoding], frame_encodings[0], tolerance=0.6)

def detect_emotion(video_frame):
    """Detects emotion from the face in the video frame."""
    emotion_data = emotion_detector.detect_emotions(video_frame)
    
    if not emotion_data:
        print("No face detected for emotion analysis.")
        return  # Return early if no face for emotion detection
    
    top_emotion = emotion_detector.top_emotion(video_frame)
    emotion, score = top_emotion
    
    print(f"Detected Emotion: {emotion} with score: {score}")
    
    suspicious_emotions = ['angry', 'disgust', 'fear', 'sad']
    if emotion in suspicious_emotions:
        print("Suspicious emotion detected. Possible cheating!")
    else:
        print("No suspicious emotion detected.")

def check_background_change(video_frame, reference_frame, threshold=50):
    """Checks if the background has significantly changed between frames."""
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_frame, gray_reference)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold

# To store the first frame as the reference frame for background comparison
reference_frame = None
reference_encoding = None

# Handle video frame events from the frontend
@socketio.on('video_data')
def handle_video_frame(data):
    global reference_frame, reference_encoding

    # Decode the base64 image data from frontend
    np_data = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if reference_frame is None:
        reference_frame = frame
        rgb_reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_reference_frame)
        if len(face_encodings) == 0:
            print("No face detected in the reference frame.")
            reference_frame = None
            return
        elif len(face_encodings) > 1:
            print("Multiple faces detected in the reference frame.")
            reference_frame = None
            return
        reference_encoding = face_encodings[0]
        print("Reference frame set.")

    try:
        face_match = compare_faces(reference_encoding, frame)
        if not face_match[0]:
            print("Face mismatch detected. Possible cheating!")

        detect_emotion(frame)

        if check_background_change(frame, reference_frame):
            print("Significant background change detected. Possible cheating!")

    except ValueError as e:
        print(f"Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
