from flask import Flask, request, jsonify, Response
from flask_cors import CORS  # Import CORS
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import os
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the model architecture
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

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

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
model = CNNModel(num_classes).to(device)
model.load_state_dict(torch.load("BabyEmotion.pth", map_location=device))
model.eval()

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#  home route
@app.route('/')
def home():
    return jsonify({"message": "Baby Emotion Recognition API is running!"})


# route  enables to predict emotions from images uploaded via POST request
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        if request.is_json:
            try:
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
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400
    else:
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please provide a clear image with a visible face.'}), 400
    
    results = []
    
    for (x, y, w, h) in faces:
        face_img = opencv_image[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(outputs, 1).item()
        
        emotion = class_names[predicted_class]
        probability = probabilities[predicted_class].item() * 100
        
        face_result = {
            'emotion': emotion,
            'probability': probability
        }
        
        results.append(face_result)
    
    return jsonify({'results': results})


# route  enables to predict emotions from video stream

@app.route('/predict_video', methods=['GET'])
def predict_video():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_class = torch.argmax(outputs, 1).item()
                
                emotion = class_names[predicted_class]
                probability = probabilities[predicted_class].item() * 100
                
                cv2.putText(frame, f'{emotion} ({probability:.2f}%)', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
