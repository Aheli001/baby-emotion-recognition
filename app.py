from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import os
import io
import google.generativeai as genai
from dotenv import load_dotenv
import onnxruntime
import threading
import queue
import time

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Face detection setup
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(face_cascade_path):
    raise RuntimeError("Haarcascade file not found!")

face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load ONNX model instead of PyTorch model
try:
    session = onnxruntime.InferenceSession("BabyEmotion.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Global variables for tracking emotion
last_emotion = None
frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory issues
processing_thread = None
stop_thread = False

def predict_emotion(img_tensor):
    """Predict emotion using ONNX model"""
    if session is None:
        return "neutral", 0.0
    
    # Convert tensor to numpy array
    input_data = img_tensor.numpy()
    
    # Get model prediction
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    
    # Process results
    probabilities = torch.nn.functional.softmax(torch.tensor(result[0]), dim=1)[0]
    predicted_class = torch.argmax(torch.tensor(result[0]), 1).item()
    
    return class_names[predicted_class], probabilities[predicted_class].item() * 100

def process_video_frames():
    """Process video frames in a separate thread"""
    global last_emotion, stop_thread
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while not stop_thread:
        success, frame = cap.read()
        if not success:
            break
            
        # Skip frames to reduce processing load
        if frame_queue.qsize() >= 2:
            continue
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detected_emotion = "neutral"
        max_probability = 0
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0)
            
            emotion, probability = predict_emotion(img_tensor)
            
            if probability > max_probability:
                max_probability = probability
                detected_emotion = emotion
            
            # Draw rectangle and emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion} ({probability:.1f}%)', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        last_emotion = detected_emotion
        
        # Compress frame before putting in queue
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_queue.put(buffer.tobytes())
        
        # Add small delay to reduce CPU usage
        time.sleep(0.1)
    
    cap.release()

@app.route('/')
def home():
    return jsonify({"message": "Baby Emotion Recognition API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            if request.is_json:
                data = request.get_json()
                if 'image' not in data:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_data = data['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                opencv_image = np.array(image)
                if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            image_file = request.files['image']
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize image if too large
        max_size = 800
        height, width = opencv_image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            opencv_image = cv2.resize(opencv_image, None, fx=scale, fy=scale)
        
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected. Please provide a clear image with a visible face.'}), 400
        
        results = []
        
        for (x, y, w, h) in faces:
            face_img = opencv_image[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0)
            
            emotion, probability = predict_emotion(img_tensor)
            
            face_result = {
                'emotion': emotion,
                'probability': probability
            }
            
            results.append(face_result)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/predict_video', methods=['GET'])
def predict_video():
    # Capture a single frame from webcam or wherever
    success, frame = video_capture.read()
    if not success:
        return "Camera error", 500

    # (Optional) do emotion detection and overlay it here
    _, buffer = cv2.imencode('.jpg', frame)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/get_current_emotion', methods=['GET'])
def get_current_emotion():
    """Endpoint to get the current detected emotion."""
    global last_emotion
    return jsonify({"emotion": last_emotion if last_emotion else "neutral"})

def get_gemini_recommendation(emotion):
    """Fetches AI-based recommendations based on emotion using Gemini API."""
    prompt = f"My baby is feeling {emotion}. Suggest a specific activity (max 25 words) that parents can do to comfort, engage, or support their baby in this state."
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            recommendation = response.text.strip()
            recommendation = recommendation.split('.')[0] + '.'
            return recommendation
        return "Try gentle rocking and soft singing to soothe the baby."
    except Exception as e:
        print(f"Error fetching Gemini API response: {str(e)}")
        return "Try gentle rocking and soft singing to soothe the baby."

@app.route('/get_suggestion/<emotion>', methods=['GET'])
def get_suggestion(emotion):
    """Endpoint to fetch suggestions from Gemini based on detected emotion."""
    recommendation = get_gemini_recommendation(emotion)
    return jsonify({"emotion": emotion, "recommendation": recommendation})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
