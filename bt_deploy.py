from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained brain tumor detection model
model = load_model("Brain_tumor_pred_new.h5")

# Define a response model to return results
class PredictionResponse(BaseModel):
    is_tumor: bool
    confidence: float

# Helper function to preprocess the image
def preprocess_image(image_data: bytes) -> np.array:
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((128, 128))  # Resize image to fit model input size
    img = img.convert("RGB")  # Convert image to RGB
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Endpoint to handle predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    
    # Preprocess image for prediction
    img_array = preprocess_image(image_data)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Assuming model output shape is [1, 1] and threshold of 0.5 for tumor detection
    confidence = float(prediction[0][0])
    is_tumor = confidence > 0.5
    
    return PredictionResponse(is_tumor=is_tumor, confidence=confidence)

# To run the FastAPI app:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
