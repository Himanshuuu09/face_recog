import cv2
import numpy as np

def check_background_change(video_frame, reference_frame, threshold=5000, blur_size=(5, 5)):
    """
    Check for significant changes in the background between the video frame and the reference frame.

    Args:
        video_frame (numpy.ndarray): The current frame from the video feed.
        reference_frame (numpy.ndarray): The reference frame (background) to compare against.
        threshold (int): The minimum number of different pixels required to consider a background change.
        blur_size (tuple): The size of the Gaussian blur kernel.

    Returns:
        bool: True if a significant change is detected, False otherwise.
    """
    # Convert frames to grayscale
    gray_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    gray_reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_video_frame = cv2.GaussianBlur(gray_video_frame, blur_size, 0)
    blurred_reference_frame = cv2.GaussianBlur(gray_reference_frame, blur_size, 0)

    # Compute the absolute difference between the frames
    difference = cv2.absdiff(blurred_video_frame, blurred_reference_frame)

    # Threshold the difference image to get binary image where significant changes are white
    _, thresholded_difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels in the thresholded difference image
    change_count = np.sum(thresholded_difference == 255)

    # Check if the count of changed pixels exceeds the threshold
    if change_count > threshold:
        return True
    else:
        return False

# Example usage
if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Read the reference frame (background)
    ret, reference_frame = cap.read()
    if not ret:
        print("Failed to capture reference frame")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Loop to process video frames
    while True:
        ret, video_frame = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break
        
        # Check for background change
        if check_background_change(video_frame, reference_frame):
            print("Background change detected!")

        # Display the video frame (for visualization purposes)
        cv2.imshow("Video Frame", video_frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
