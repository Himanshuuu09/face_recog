from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit,disconnect
import os
import time
import jwt
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import face_recognition
from collections import defaultdict, Counter
import queue
import dlib
from PIL import Image
import shutil
import uuid
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'omrscanner'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)
ENDPOINT_URL = "https://4wq09l1k-5290.inc1.devtunnels.ms/AddFaceData"
# Initialize dictionaries and variables
client_info = {}
video_queues = defaultdict(queue.Queue)
active_connections = 0  # Track number of active connections

# Load ML models with error handling
try:
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/OMR-09/Desktop/project_directory/haarcascade_frontalface_default.xml")
    emotion_classifier = load_model(
        "C:/Users/OMR-09/Desktop/project_directory/model.h5")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    KNOWN_IMAGES_DIR = "C:/Users/OMR-09/Desktop/project_directory/NEW1"  # Change this to the directory of your known person images
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/OMR-09/Desktop/project_directory/shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Function to load known face encodings
def load_known_faces(known_images_dir):
    known_encodings = []
    for file_name in os.listdir(known_images_dir):
        image_path = os.path.join(known_images_dir, file_name)
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    return known_encodings

known_face_encodings = load_known_faces(KNOWN_IMAGES_DIR)

# Function to create user-specific video directory
def create_user_video_dir(email):
    user_video_dir = os.path.join(os.path.dirname(__file__), 'videos', email)
    if not os.path.exists(user_video_dir):
        os.makedirs(user_video_dir)
    return user_video_dir

# Function to create file write stream
def create_file_write_stream(file_name, user_video_dir):
    try:
        file_path = os.path.join(user_video_dir, f"{file_name}.webm")
        file_stream = open(file_path, 'wb')
        return file_stream, file_path
    except Exception as e:
        print(f"Error creating file write stream: {e}")
        return None, None

# Function to detect emotions from video frames
def emotion_fdetect(video_path, user_video_dir, condition_counters):
    emotion_counter = Counter()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Video Capture Error"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            save_frame(frame, "no_face_detected", user_video_dir, condition_counters)
            condition_counters['no_face_detected'] += 1

        if len(faces) > 1:
            save_frame(frame, "multiple_faces_detected", user_video_dir, condition_counters)
            condition_counters['multiple_faces_detected'] += 1

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_counter[label] += 1

    cap.release()
    most_common_emotion = emotion_counter.most_common(1)
    if most_common_emotion:
        return most_common_emotion[0][0]
    else:
        return "No emotions detected."

# Function to detect if a person matches a known face
def detect_person_match(video_path, user_video_dir, condition_counters):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Video Capture Error"

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Video Capture Error"

    face_locations = face_recognition.face_locations(frame)
    face_count = len(face_locations)

    if face_count == 0:
        save_frame(frame, "no_face_detected", user_video_dir, condition_counters)
        condition_counters['no_face_detected'] += 1
        cap.release()
        return "No Face Detected"

    if face_count > 1:
        save_frame(frame, "multiple_faces_detected", user_video_dir, condition_counters)
        condition_counters['multiple_faces_detected'] += 1
        cap.release()
        return "More than One Face Detected"

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for known_encoding in known_face_encodings:
        match = face_recognition.compare_faces([known_encoding], face_encodings[0])
        if match[0]:
            neck_bending = detect_neck_bending(frame, face_locations[0])
            cap.release()
            if neck_bending:
                save_frame(frame, "neck_bending", user_video_dir, condition_counters)
                condition_counters['neck_bending'] += 1
                return "Neck Movement"
            else:
                return "Match"

    # Save frame when no match found
    save_frame(frame, "not_match", user_video_dir, condition_counters)
    condition_counters['not_match'] += 1

    cap.release()
    return "Not Match"


def detect_neck_bending(frame, face_location):
    face_landmarks = face_recognition.face_landmarks(frame, [face_location])
    if not face_landmarks:
        return False

    face_landmarks = face_landmarks[0]
    top_nose = face_landmarks['nose_bridge'][0]
    bottom_nose = face_landmarks['nose_tip'][0]
    top_chin = face_landmarks['chin'][8]
    bottom_chin = face_landmarks['chin'][0]

    neck_vector = np.array(bottom_chin) - np.array(top_chin)
    face_vector = np.array(bottom_nose) - np.array(top_nose)

    angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
                                  (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
    print(angle)
    return angle > 132 or angle < 124

# Function to detect if a person is wearing glasses
def detect_glasses(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(img)
    if len(faces) == 0:
        return "No Face Detected"

    rect = faces[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    nose_bridge_x = [landmarks[i][0] for i in [28, 29, 30, 31, 33, 34, 35]]
    nose_bridge_y = [landmarks[i][1] for i in [28, 29, 30, 31, 33, 34, 35]]

    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    y_min = landmarks[20][1]
    y_max = landmarks[31][1]

    img2 = Image.fromarray(img)
    img2 = img2.crop((x_min, y_min, x_max, y_max))

    img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=90, threshold2=185)

    edges_center = edges.T[int(len(edges.T) / 2)]

    if 255 in edges_center:
        return "Glasses Present"
    else:
        return "Glasses Absent"

# Function to detect background movement in a video
def detect_background_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Video Capture Error"

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return "Video Capture Error"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    movement_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        movement = cv2.countNonZero(thresh)

        if movement > 2500:  # Threshold for detecting movement
            movement_detected = True
            break

        prev_gray = gray

    cap.release()
    if movement_detected:
        return "Background Movement Detected"
    else:
        return "No Background Movement"

# Function to handle client connection
@socketio.on('connect')
def handle_connect():
    print("connected successfully")
    token = request.args.get('token')
    if token:
        print(f'Received token: {token}')
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            print(f'Client connected and authenticated: {payload}')
            
            # Extract email from the payload
            email = payload.get('email')  # Assuming 'email' is a key in your JWT payload
            print(email)
            examid=payload.get('examId')
            print(examid)
            orgId=payload.get('orgId')
            print(orgId)
            
            if email:
                print(f'Client email: {email}')
                # You can add more logic here to handle the authenticated user
            else:
                print('Email not found in token payload')
                disconnect()
                
        except jwt.ExpiredSignatureError:
            print('Token has expired')
            disconnect()
        except jwt.InvalidTokenError:
            print('Invalid token')
            disconnect()
    else:
        print('Token not received')
        disconnect()
                
    global active_connections
    active_connections += 1
    print("Client Connection established")
    sid = request.sid
    print(sid)
    client_info[sid] = {}
    client_info[sid]['email'] = email  # Store email instead of username
    client_info[sid]['orgId'] = orgId # Store orgId instead of
    client_info[sid]['examid']=examid
    client_info[sid]['video_dir'] = create_user_video_dir(email)
    client_info[sid]['condition_counters'] = Counter()

    # Emit initial condition counters to the client
    emit_condition_counters(sid, email) 

# Function to handle incoming video data
# Function to handle incoming video data
@socketio.on('video_data')
def handle_video_data(data):
    print(data)
    sid = request.sid
    email = client_info[sid]['email']  # Retrieve email from client_info
    video_dir = client_info[sid]['video_dir']
    condition_counters = client_info[sid]['condition_counters']

    file_name = f"video_{int(time.time() * 1000)}"
    file_stream, file_path = create_file_write_stream(file_name, video_dir)
    emit('video_stream', data, room=sid)
    if file_stream is None:
        emit('result', {'email': email, 'result': 'File Write Error', 'emotion': 'N/A', 'glasses': 'N/A', 'background_movement': 'N/A'}, room=sid)
        return

    try:
        file_stream.write(data)
    except Exception as e:
        print(f"Error writing to file: {e}")
        emit('result', {'email': email, 'result': 'File Write Error', 'emotion': 'N/A', 'glasses': 'N/A', 'background_movement': 'N/A'}, room=sid)
    finally:
        file_stream.close()
        video_queues[sid].put(file_path)  # Put the file path into the client's queue
    
    # If the queue has only one video, process it
    if video_queues[sid].qsize() == 1:
        process_next_video(sid,condition_counters)
        
        # Endpoint to get condition counters in real-time
# Endpoint to get condition counters in real-time
@app.route('/condition_counters', methods=['POST'])
def get_condition_counters():
    data = request.get_json()  # Assuming data is sent in JSON format
    sid = data.get('sid')  # Get sid from JSON payload

    if sid in client_info:
        condition_counters = client_info[sid]['condition_counters']
        relevant_counters = {
            'No Face Detected': condition_counters.get('no_face_detected', 0),
            'Not Match': condition_counters.get('not_match', 0),
            'Neck Bending': condition_counters.get('neck_bending', 0),
            'Multiple Faces Detected': condition_counters.get('multiple_faces_detected', 0)
        }
        warning_count = sum(relevant_counters.values())
        return jsonify({'success': True, 'counters': relevant_counters,'warningCount': warning_count}), 200
    else:
        return jsonify({'success': False, 'message': 'Client not found'}), 404
    

# Function to process the next video in queue
# Function to process the next video in queue   
def process_next_video(sid,condition_counters):
    email = client_info[sid]['email']  # Retrieve email from client_info
    orgId = client_info[sid]['orgId']
    examid = client_info[sid]['examid']

    file_path = video_queues[sid].get()  # Get the next video file path from the queue
    result = detect_person_match(file_path, client_info[sid]['video_dir'], client_info[sid]['condition_counters'])
    emotion = emotion_fdetect(file_path, client_info[sid]['video_dir'], client_info[sid]['condition_counters'])
    glasses = detect_glasses(cv2.VideoCapture(file_path).read()[1])
    background_movement = detect_background_movement(file_path)
    
    emit('result', {'email': email, 'result': result, 'emotion': emotion, 'glasses': glasses, 'background_movement': background_movement}, room=sid)
    send_data_to_endpoint(email, examid, orgId, condition_counters)
    # Emit condition counters update to the client
    emit_condition_counters(sid, email)
   
    # If there are more videos in the queue, process the next one
    if not video_queues[sid].empty():
        process_next_video(sid)
    
    
    
# Function to emit condition counters update to the client
def emit_condition_counters(sid, email):
    counters = client_info[sid]['condition_counters']
    emit('condition_counters', {'email': email, 'counters': counters}, room=sid)


# Function to handle client disconnection
@socketio.on('disconnect')
def handle_disconnect():
    global active_connections
    sid = request.sid
    if sid in client_info:
        username = f"user_{sid}"
        condition_counters = client_info[sid]['condition_counters']
        active_connections -= 1
        del client_info[sid]

        # Print condition counters
        print(f"Counters for {username}: {dict(condition_counters)}")

        # Delete user-specific video directory if it exists and no other active sessions are using it
        user_video_dir = os.path.join(os.path.dirname(__file__), 'videos', username)
        if os.path.exists(user_video_dir) and not any(client_info[sid]['video_dir'] == user_video_dir for sid in client_info if 'video_dir' in client_info[sid] and client_info[sid]['video_dir'] == user_video_dir):
            try:
                shutil.rmtree(user_video_dir)
                print(f"Deleted user video directory: {user_video_dir}")
            except Exception as e:
                print(f"Error deleting user video directory: {e}")

        emit('user_disconnected', {'username': username}, broadcast=True)

# Function to save frames based on conditions
# Function to save frames based on conditions
def save_frame(frame, condition, user_video_dir, condition_counters):
    save_dir = os.path.join(user_video_dir, "images")  # Save frames under "images" directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate unique file name
    file_name = f"{condition}_{str(uuid.uuid4())[:8]}.jpg"
    file_path = os.path.join(save_dir, file_name)
    cv2.imwrite(file_path, frame)

    # Increment condition counters
    if condition == "no_face_detected":
        condition_counters['No Face Detected'] += 1
    elif condition == "not_match":
        condition_counters['Not Match'] += 1
    elif condition == "neck_bending":
        condition_counters['Neck Bending'] += 1
    elif condition == "multiple_faces_detected":
        condition_counters['Multiple Faces Detected'] += 1



def send_data_to_endpoint(email, examid, orgid, condition_counters):
    endpoint = "https://4wq09l1k-5290.inc1.devtunnels.ms/AddFaceData"
    
    data = {
        "examId": examid,
        "userEmail": email,
        "multipleFaceDetected": condition_counters.get('multiple_faces_detected', 0),
        "neckBending": condition_counters.get('neck_bending', 0),
        "noFaceDetected": condition_counters.get('no_face_detected', 0),
        "noMatch": condition_counters.get('not_match', 0),
        "warningCount": sum(condition_counters.values()),  # Total warning count
        "orgId": orgid
    }
    
    try:
        response = requests.post(endpoint, json=data)
        response.raise_for_status()  # Raise exception for bad response status
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print(f"Unexpected response from server: {response.status_code} - {response.text}")        
        # print("Data sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {e}")


if __name__ == '__main__':
    socketio.run(app, port=4500, host='0.0.0.0', debug=True)
