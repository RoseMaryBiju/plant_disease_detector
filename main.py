import os
import json
import requests
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set up page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    model_url = "YOUR_DIRECT_DOWNLOAD_LINK"  # Replace with your actual download link
    model_path = "downloaded_model.h5"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... This may take a while."):
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
    
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Load class indices
class_indices_path = "class_indices.json"  # Make sure this file is in your repository
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error(f"Class indices file not found at {class_indices_path}. Please ensure the file exists in the correct location.")
    st.stop()

# Reverse the class indices dictionary
class_names = {v: k for k, v in class_indices.items()}

def preprocess_image(image):
    img = image.resize((224, 224))  # Adjust size as needed
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Plant Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Disease"):
        disease, confidence = predict_disease(image)
        st.success(f"Predicted Disease: {disease}")
        st.info(f"Confidence: {confidence:.2f}%")

# Add some information about the app
st.markdown("""
    ## About this app
    This app uses a deep learning model to classify plant diseases based on leaf images.
    Upload an image of a plant leaf, and the app will predict if the plant is healthy or identify the disease.
""")
