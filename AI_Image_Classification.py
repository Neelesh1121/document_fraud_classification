import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import logging

# ------------------- Configure Logger -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------- Constants & Paths -------------------
img_size = 48  # Target image size for the model
path = '/content/AI-Image-Classifier/Testing/ai1.png'
model_path = '/content/AI-Image-Classifier/AIGeneratedModel.h5'

# ------------------- Load Pretrained Model -------------------
try:
    model = tf.keras.models.load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load model.")
    raise

# ------------------- Load and Preprocess Image -------------------
try:
    image = Image.open(path)
    logging.info(f"Image loaded from path: {path}")
    
    # Convert to RGB to match input shape
    image = image.convert('RGB')
    
    # Resize and crop to (48x48)
    image = ImageOps.fit(image, (img_size, img_size), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    img_array = img_to_array(image)
    
    # Normalize pixel values to [0,1]
    normalized_array = img_array / 255.0
    
    # Prepare input for model (add batch dimension)
    input_array = np.expand_dims(normalized_array, axis=0)
    logging.info("Image preprocessed successfully.")
except Exception as e:
    logging.exception("Image preprocessing failed.")
    raise

# ------------------- Run Prediction -------------------
try:
    prediction = model.predict(input_array)
    predicted_value = prediction[0][0]

    logging.info(f"Model prediction score: {predicted_value:.4f}")

    if predicted_value <= 0.5:
        print("The given image is Real.")
    else:
        print("The given image is AI Generated.")
except Exception as e:
    logging.exception("Model prediction failed.")
    raise
