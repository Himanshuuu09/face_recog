from flask import Flask, request, render_template, jsonify, redirect, url_for
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS 
from datetime import datetime
from flask import render_template
import subprocess
import os
import face_recognition


app = Flask(__name__)
CORS(app)

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
@app.route('/photo_page', methods=['GET'])
def photo_page():
    return render_template('photo_upload.html') 

def preprocess_image(image):
    """Preprocess the image to enhance face detection in dim lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return enhanced

def detect_face(image):
    """Detects face and returns True if exactly one face is detected, otherwise False."""
    image = preprocess_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    return len(face_locations) == 1


@app.route('/upload', methods=['POST'])
def upload_image():
    global filename
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])

        # Open the image and save it to a file
        image = Image.open(BytesIO(image_data))

        # Ensure the 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Check for face detection
        if not detect_face(image):
            return jsonify({"message": "Error: No face detected in the image."}), 400

        # Save the image if face is detected
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'uploads/photo_{timestamp}.jpg'
        image.save(f'{filename}')

        return jsonify({"message": "Image uploaded and face detected successfully.", "redirect": "/success"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/success', methods=['GET'])
def success_page():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)