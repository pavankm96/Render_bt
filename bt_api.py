from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

# Load your model
model = load_model("model.h5")

# Preprocess the uploaded image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize to match the input shape of your model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image
    img = Image.open(file.file)
    
    # Preprocess it
    processed_img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Determine result (assuming binary classification: 0=Non-malignant, 1=Malignant)
    result = "Malignant" if prediction[0][0] > 0.5 else "Non-malignant"
    
    return {"result": result}
