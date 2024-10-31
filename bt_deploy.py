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
        # Load the image directly
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Save the processed image to a temporary file-like object
        image_byte_array = io.BytesIO()
        Image.fromarray((processed_image[0] * 255).astype(np.uint8)).save(image_byte_array, format='PNG')
        image_byte_array.seek(0)  # Move the cursor to the beginning of the file-like object

        # Send request to Hugging Face API
        response = requests.post(
            API_URL,
            headers=headers,
            files={"file": ("image.png", image_byte_array, "image/png")}  # Sending image as a file
        )

        # Check for errors in the response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
        # Get and return the model prediction
        prediction = response.json()
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

