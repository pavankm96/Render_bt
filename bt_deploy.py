from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
from PIL import Image
import io
import numpy as np
import os

app = FastAPI()

# Hugging Face API details
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/pavankm96/brain_tumor_det"
API_TOKEN = "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"

# Set up headers with authentication token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Preprocess image (modify as needed based on model requirements)
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Example size, adjust to model input size
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array.tolist()  # Convert to list for JSON serialization

# Define prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Send request to Hugging Face API
        response = requests.post(
            HUGGING_FACE_API_URL,
            headers=headers,
            json={"inputs": processed_image}
        )
        
        # Check for errors in the response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
        # Get and return the model prediction
        prediction = response.json()
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
