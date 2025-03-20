from flask import Flask, request, jsonify
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

# Define the model architecture (same as in your original code)
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
            nn.Linear(256, num_classes)  # Output layer
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


# Define the same transformations used in training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # Update this based on your actual number of classes
model = CNNModel(num_classes).to(device)
model.load_state_dict(torch.load("BabyEmotion.pth", map_location=device))
model.eval()

# Get class names (update these based on your actual classes)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def home():
    return jsonify({"message": "Baby Emotion Recognition API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        # Check if image is sent as base64 in JSON
        if request.is_json:
            try:
                data = request.get_json()
                if 'image' not in data:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Decode base64 image
                image_data = data['image']
                if image_data.startswith('data:image'):
                    # Handle data URL format
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert PIL Image to OpenCV format
                opencv_image = np.array(image)
                if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400
    else:
        # Handle file upload
        image_file = request.files['image']
        # Read image file
        image_bytes = image_file.read()
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = opencv_image[y:y+h, x:x+w]
        
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(outputs, 1).item()
        
        # Get the class name and probability
        emotion = class_names[predicted_class]
        probability = probabilities[predicted_class].item() * 100
        
        # Create a result object
        face_result = {
            'emotion': emotion,
            'probability': probability,
            'face_position': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'all_probabilities': {
                class_names[i]: float(probabilities[i].item() * 100) 
                for i in range(len(class_names))
            }
        }
        
        results.append(face_result)
    
    if not results:
        return jsonify({
            'error': 'No faces detected in the image',
            'faces_detected': 0
        }), 200
    
    return jsonify({
        'faces_detected': len(results),
        'results': results
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(debug=True, host='0.0.0.0', port=port)