from fastapi import FastAPI, File, UploadFile, HTTPException
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io

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
