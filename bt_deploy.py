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

# Preprocess the image (modify to match model's expected input format)
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to model's input size, if known
    image_array = np.array(image) / 255.0  # Normalize to 0-1 range
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array.tolist()  # Convert to list for JSON serialization

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Send the processed image to the Hugging Face model
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": processed_image}
        )

        # Check if the response was successful
        if response.status_code == 200:
            prediction = response.json()  # Get the model's prediction
            return {"prediction": prediction}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app = FastAPI()

# Define your model repository and filename
model_name = "pavankm96/brain_tumor_det"  # Replace with your actual Hugging Face model name
model_filename = "Brain_tumor_pred.h5"   # Replace with the actual filename of your .h5 model

# Download and load the model
try:
    model_path = hf_hub_download(repo_id=model_name, filename=model_filename)
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to the model's expected input size
    image_array = np.array(image) / 255.0  # Normalize to 0-1
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Process prediction (assuming binary classification: 0 for non-tumor, 1 for tumor)
        tumor_detected = bool(np.argmax(prediction))  # Adjust based on your model's output structure
        return {"tumor_detected": tumor_detected, "confidence": float(prediction[0][np.argmax(prediction)])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
