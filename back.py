import cv2
import face_recognition
import numpy as np
from fer import FER

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
    return face_encodings[0]

def compare_faces(image_encoding, video_frame):
    """Compares the face in the image with the face in a video frame."""
    frame = preprocess_image(video_frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_encodings = face_recognition.face_encodings(rgb_frame)
    
    if len(frame_encodings) == 0:
        raise ValueError("No face detected in the video frame.")
    elif len(frame_encodings) > 1:
        raise ValueError("More than one person detected in the video. Possible cheating detected.")
    
    return face_recognition.compare_faces([image_encoding], frame_encodings[0], tolerance=0.6)

def check_background_change(video_frame, reference_frame, motion_threshold=0.05, area_threshold=2000):
    """Checks if the background has significantly changed."""
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Blur images to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    blurred_reference = cv2.GaussianBlur(gray_reference, (21, 21), 0)

    # Compute the absolute difference between the reference and current frames
    frame_diff = cv2.absdiff(blurred_reference, blurred_frame)

    # Threshold the difference to filter out small changes
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in small gaps
    dilated_thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track significant motion
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue  # Ignore small areas

        # Draw bounding box around the detected motion (for visualization)
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Significant motion detected
        motion_detected = True

    # Calculate the percentage of the frame covered by motion
    motion_area_percentage = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
    print(f"Motion area percentage: {motion_area_percentage * 100:.2f}%")

    # Return True if the motion is above the motion_threshold
    return motion_detected and motion_area_percentage > motion_threshold

def analyze_expression(video_frame, emotion_threshold=0.5):
    """Analyzes facial expressions using FER and detects suspicious behavior."""
    detector = FER()
    frame = preprocess_image(video_frame)
    emotions = detector.detect_emotions(frame)
    
    if emotions:
        emotion = emotions[0]["emotions"]
        max_emotion = max(emotion, key=emotion.get)
        max_score = emotion[max_emotion]
        
        print(f"Detected emotion: {max_emotion}, Score: {max_score}")
        
        if max_score > emotion_threshold and max_emotion in ['fear', 'surprise', 'angry', 'disgust']:
            print("Suspicious emotion detected, possible cheating!")
            return False
    return True

def process_live_video(image_encoding, background_threshold=0.05, area_threshold=2000, emotion_threshold=0.5):
    """Capture live video and perform face detection and other checks."""
    cap = cv2.VideoCapture(0)
    ret, reference_frame = cap.read()
    if not ret:
        print("Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Compare face in live video with uploaded image
            face_match = compare_faces(image_encoding, frame)
            if not face_match[0]:
                print("Face mismatch detected. Possible cheating!")
                break

            # Detect large changes in the background
            if check_background_change(frame, reference_frame, background_threshold, area_threshold):
                print("Significant background change detected. Possible cheating!")
                break
            
            # Analyze facial expression for suspicious activity
            if not analyze_expression(frame, emotion_threshold):
                print("Cheating detected based on facial expression!")
                break

            # Display the resulting frame
            cv2.imshow('Live Video Feed', frame)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except ValueError as e:
            print(f"Error: {e}")
            continue  # Continue processing other frames

    cap.release()
    cv2.destroyAllWindows()

def main(image_path, background_threshold=0.05, area_threshold=2000, emotion_threshold=0.5):
    """Main function to start the live video processing."""
    image = face_recognition.load_image_file(image_path)
    
    try:
        image_encoding = detect_face(image)
    except ValueError as e:
        print(e)
        return
    
    process_live_video(image_encoding, background_threshold, area_threshold, emotion_threshold)

if __name__ == "__main__":
    image_path = r"C:\path\to\image.jpg"
    main(image_path)
