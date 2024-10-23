from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model('model.h5')  # Load your model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((224, 224))  # Resize as required by your model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    label = 'Tumor' if prediction[0] > 0.5 else 'No Tumor'  # Adjust according to your model's output

    return JSONResponse(content={"prediction": label})
