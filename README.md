🧑‍🔬 Ethnicity Classification API
This is a FastAPI-based RESTful API for classifying the ethnicity of a person from an uploaded image using a deep learning model. The model identifies one of four predefined ethnicity categories.

📌 Features
Upload an image (JPG/PNG)

Face detection and ethnicity classification

Returns ethnicity label, confidence score, and probability distribution

Saves uploaded images for traceability

🧠 Ethnicity Labels
The model predicts the following ethnic groups:

makefile
Copy
Edit
0: Black  
1: South East Asian  
2: Indian  
3: White
🚀 Getting Started
Prerequisites
Python 3.8+

pip

Installation
Clone the repository

bash
Copy
Edit
pip install -r requirements.txt
Project structure

bash
Copy
Edit
├── app.py                 # FastAPI app
├── model.py                # Model architecture (EthnicityModel)
├── inference.py            # classify_ethnicity() function
├── Model/
│   └── refined_v1_epoch_22.pth  # Pretrained model weights
├── detected_images/        # Folder to save uploaded images
└── requirements.txt        # Python dependencies
Run the server

bash
Copy
Edit
uvicorn app:app --reload
Access the API Docs

Open your browser and go to: http://127.0.0.1:8000/docs

📥 API Endpoint
POST /predict/
Request
Body: Multipart form with an image file (image/jpeg or image/png)

Response (JSON)
json
Copy
Edit
{
  "ethnicity": "Indian",
  "confidence": 0.92,
  "face_confidence": 0.88,
  "all_probabilities": {
    "Black": 0.02,
    "South East Asian": 0.03,
    "Indian": 0.92,
    "White": 0.03
  },
  "saved_path": "detected_images/20250718_124500_test.jpg"
}
Errors
400 Invalid file type

400 Could not decode image

400 Face not detected or prediction failed

🛠️ Model
Model architecture defined in model.py (likely a CNN)

Trained weights loaded from Model/refined_v1_epoch_22.pth

📂 Output
All uploaded images are saved in the detected_images/ folder for auditing and debugging.
