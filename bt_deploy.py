from fastapi import FastAPI, File, UploadFile, HTTPException
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Hugging Face API URL and token
API_URL = "https://api-inference.huggingface.co/models/pavankm96/brain_tumor_det"
API_TOKEN = "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"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Preprocess the image (modify based on your model's requirements)
def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # Resize to the model's input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array  # Return the processed image array directly

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Send request to Hugging Face API
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": processed_image.tolist()}  # Sending image as JSON input
        )

        # Check for errors in the response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
        # Get and return the model prediction
        prediction = response.json()
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
