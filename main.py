from flask import Flask, request, jsonify
import cv2
import base64
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

import os
os.environ['SSL_CERT_FILE'] = 'DevHost24/etc/ssl/certs/ca-certificates.crt'  # Adjust the path as necessary for your system



# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained fraud detection model
with open('finall.pkl', 'rb') as f:
    fraud_detection_model = pickle.load(f)

# MongoDB Atlas connection string
mongo_uri = 'mongodb+srv://aksh9881:aksh9881@dev.zqib6ic.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true&appName=Dev'


client = MongoClient(mongo_uri)

# Specify the database and collection
db = client.get_database('fraud_detection')  # Replace with your database name
collection = db.get_collection('transactions')  # Replace with your collection name

# Function to encode faces
def encode_faces(faces):
    encoded_faces = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (100, 100))
        encoded_faces.append(resized_face.flatten())
    return encoded_faces

# Function to compare faces using cosine similarity
def compare_faces(face1, face2):
    return cosine_similarity([face1], [face2])[0][0]

# Route for storing face
@app.route('/store_face', methods=['POST'])
def store_face():
    try:
        if request.content_type != 'application/json':
            app.logger.error("Content-Type is not application/json")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        
        if 'image' not in data:
            app.logger.error("Missing image data")
            return jsonify({'error': 'Missing image data'}), 400
        
        image_data = data['image']
        
        app.logger.info("Received image data")
        
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            app.logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        app.logger.info(f"Faces detected: {len(faces)}")
        
        if len(faces) == 0:
            app.logger.warning("No face detected")
            return jsonify({'error': 'No face detected'}), 400
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        data_to_insert = {
            'image_base64': img_base64,
            'faces_detected': len(faces)
        }
        
        inserted_result = collection.insert_one(data_to_insert)
        
        app.logger.info(f"Inserted document id: {inserted_result.inserted_id}")
        
        return jsonify({
            'image': img_base64,
            'faces_detected': len(faces),
            'mongo_inserted_id': str(inserted_result.inserted_id)
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred processing your request'}), 500
    
@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    try:
        if request.content_type != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        
        if 'image' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing image data or user ID'}), 400
        
        image_data = data['image']
        user_id = data['user_id']
        
        # Fetch stored image from the database
        stored_data = collection.find_one({'user_id': user_id})
        
        if not stored_data:
            return jsonify({'error': 'User not found'}), 404
        
        stored_image_base64 = stored_data['image_base64']
        stored_image_data = base64.b64decode(stored_image_base64)
        stored_image_np = np.frombuffer(stored_image_data, np.uint8)
        stored_img = cv2.imdecode(stored_image_np, cv2.IMREAD_COLOR)
        
        # Decode the current image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        current_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if current_img is None or stored_img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Convert images to grayscale
        stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the images
        stored_faces = face_cascade.detectMultiScale(stored_gray, scaleFactor=1.3, minNeighbors=5)
        current_faces = face_cascade.detectMultiScale(current_gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(stored_faces) == 0 or len(current_faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        # Assuming one face per image for simplicity
        x, y, w, h = stored_faces[0]
        stored_face = stored_gray[y:y+h, x:x+w]
        
        x, y, w, h = current_faces[0]
        current_face = current_gray[y:y+h, x:x+w]
        
        # Resize faces to the same size
        stored_face_resized = cv2.resize(stored_face, (200, 200))
        current_face_resized = cv2.resize(current_face, (200, 200))
        
        # Compare the faces (using Mean Squared Error for simplicity, can be replaced with more advanced methods)
        error = np.mean((stored_face_resized - current_face_resized) ** 2)
        threshold = 1000  # You may need to adjust this threshold
        
        if error < threshold:
            return jsonify({'message': 'Transaction approved'}), 200
        else:
            return jsonify({'message': 'Transaction denied, faces do not match'}), 403
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500


# Route for payment processing
@app.route('/process_payment', methods=['POST'])
def process_payment():
    # Ensure content type is JSON
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    # Receive JSON data
    data = request.json

    # Check if 'image' and 'amount' keys exist in JSON data
    if 'image' not in data or 'amount' not in data:
        return jsonify({'error': 'Missing image or amount data'}), 400

    # Decode base64 image data
    image_data = data['image']
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    # Encode detected faces
    encoded_faces = encode_faces(faces)

    # Retrieve stored faces from the database
    stored_faces = collection.find()

    for record in stored_faces:
        stored_face_data = base64.b64decode(record['image_base64'])
        stored_face_array = np.frombuffer(stored_face_data, np.uint8)
        stored_face = cv2.imdecode(stored_face_array, cv2.IMREAD_GRAYSCALE)
        stored_face_encoded = stored_face.flatten()

        # Compare detected face with stored face
        for encoded_face in encoded_faces:
            similarity = compare_faces(encoded_face, stored_face_encoded)
            if similarity > 0.8:  # Assuming a threshold of 0.8 for face match
                # Process the payment if faces match
                amount = data['amount']
                return jsonify({'message': f'Payment of {amount} processed successfully', 'similarity': similarity}), 200

    return jsonify({'error': 'Face mismatch. Payment cannot be processed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
