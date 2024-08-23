import cv2
import face_recognition
import numpy as np

def preprocess_image(image):
    """Preprocess the image to enhance face detection in dim lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return enhanced

def detect_face(image):
    """Detects face and returns the face encoding."""
    image = preprocess_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 0:
        raise ValueError("No face detected in the image.")
    elif len(face_locations) > 1:
        raise ValueError("More than one face detected in the image.")
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings[0], face_locations[0]

def compare_faces(image_encoding, video_frame):
    """Compares the face in the image with the face in a video frame."""
    frame = preprocess_image(video_frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_encodings = face_recognition.face_encodings(rgb_frame)
    
    if len(frame_encodings) == 0:
        raise ValueError("No face detected in the video frame.")
    elif len(frame_encodings) > 1:
        raise ValueError("More than one person detected in the video. Possible cheating detected.")
    
    print("Face matched and no other person found")
    return face_recognition.compare_faces([image_encoding], frame_encodings[0], tolerance=0.6)

def check_background_change(video_frame, reference_frame, threshold=50):
    """Checks if the background has significantly changed between frames."""
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_frame, gray_reference)
    mean_diff = np.mean(diff)
    # print(f"Background difference mean: {mean_diff}")
    print("background is safe")
    return mean_diff > threshold

def detect_facial_landmarks(frame):
    """Detect facial landmarks to help determine head pose."""
    face_landmarks_list = face_recognition.face_landmarks(frame)
    if len(face_landmarks_list) == 0:
        raise ValueError("No face landmarks detected.")
    
    return face_landmarks_list[0]  # Return the landmarks for the first detected face

def is_looking_down(landmarks, frame):
    """Determine if the person is looking downward."""
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate the center of the eyes
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    
    # Calculate the angle of the line between the centers of the eyes and the horizontal axis
    eye_line = right_eye_center - left_eye_center
    eye_angle = np.arctan2(eye_line[1], eye_line[0]) * (180.0 / np.pi)
    
    # Assuming a downward angle threshold to indicate looking down
    downward_threshold = 30  # degrees
    return abs(eye_angle) > downward_threshold

def process_live_video(image_encoding, background_threshold=50):
    """Capture live video and perform face detection, background change check, and posture check."""
    cap = cv2.VideoCapture(0)
    ret, reference_frame = cap.read()
    if not ret:
        print("Could not access the webcam.")
        return

    background_change_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Detect face and facial landmarks
            face_encoding, face_location = detect_face(frame)
            face_match = compare_faces(image_encoding, frame)
            if not face_match[0]:
                print("Face mismatch detected. Possible cheating!")
                break

            if check_background_change(frame, reference_frame, background_threshold):
                print("Significant background change detected. Possible cheating!")
                background_change_detected = True

            # Detect facial landmarks and check posture
            landmarks = detect_facial_landmarks(frame)
            if is_looking_down(landmarks, frame):
                print("Person is looking downward. Maintain proper posture!")
                break

            # Display the resulting frame
            cv2.imshow('Live Video Feed', frame)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except ValueError as e:
            print(f"Error: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()
    
    return not background_change_detected

def main(image_path, background_threshold=50):
    """Main function to start the live video processing."""
    image = face_recognition.load_image_file(image_path)
    
    try:
        image_encoding, _ = detect_face(image)
    except ValueError as e:
        print(e)
        return
    
    if process_live_video(image_encoding, background_threshold):
        print("No cheating detected.")
    else:
        print("Cheating detected!")

if __name__ == "__main__":
    image_path = r"C:\Users\OMR-09\Pictures\Camera Roll\WIN_20240823_10_09_20_Pro.jpg"  # Path to the uploaded image
    background_threshold = 50  # Adjustable threshold for background change
    main(image_path, background_threshold)
