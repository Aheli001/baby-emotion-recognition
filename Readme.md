# Baby Emotion Recognition System

## Project Overview

The Baby Emotion Recognition System is an image classification model designed to identify and classify the emotional state of a baby based on facial expressions. The system uses a Convolutional Neural Network (CNN) to analyze grayscale images of babies' faces and predict emotions such as anger, disgust, fear, happiness, neutrality, sadness, and surprise.

This project aims to provide a reliable and fast emotion recognition system that can be used in various applications, including monitoring baby emotions for medical, parenting, or entertainment purposes.

## Project Structure

```
baby-emotion-recognition/
├── app.py              # Main Flask application
├── train.ipynb         # Model training notebook
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── BabyEmotion.onnx   # Exported ONNX model
├── BabyEmotion.pth    # PyTorch model weights
├── uploads/           # Directory for uploaded images
└── images/            # Directory for sample images
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd baby-emotion-recognition
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   - Create a `.env` file in the root directory
   - Add necessary environment variables (if required)

## Usage

1. **Running the Application**
   ```bash
   python app.py
   ```
   The application will start on `http://localhost:5000`

2. **Model Training**
   - Open `train.ipynb` in Jupyter Notebook
   - Follow the notebook cells to train the model
   - The trained model will be saved as `BabyEmotion.pth`
   - Export to ONNX format for deployment

3. **Making Predictions**
   - Upload baby face images through the web interface
   - The system will analyze the image and return emotion predictions
   - Results include confidence scores for each emotion

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: Grayscale images of baby faces
- Output: Emotion classification (7 classes)
- Supported Emotions:
  - Anger
  - Disgust
  - Fear
  - Happiness
  - Neutrality
  - Sadness
  - Surprise

## Dependencies

- Flask: Web application framework
- PyTorch: Deep learning framework
- OpenCV: Image processing
- ONNX: Model deployment
- Other dependencies listed in requirements.txt

