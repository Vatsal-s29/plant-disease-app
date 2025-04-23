import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import (
    densenet,
    mobilenet_v2,
    efficientnet
)

# ✅ Streamlit Page Config
st.set_page_config(page_title="🌿 Enhanced Plant Disease Classifier", layout="centered")

# Parameters
IMG_SIZE = (128, 128)
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy"
]

# Preprocessing functions
efficientnet_preprocess = efficientnet.preprocess_input
mobilenet_preprocess = mobilenet_v2.preprocess_input
densenet_preprocess = densenet.preprocess_input

# Load Models (cache to avoid reloading)
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("efficientnetb0_final.h5")
    model2 = tf.keras.models.load_model("mobilenetv2_finetuned.h5")
    model3 = tf.keras.models.load_model("densenet121_finetuned.h5")
    return model1, model2, model3

model1, model2, model3 = load_models()

# Image Preprocessing for each model
def preprocess_image_for_model(image, preprocess_fn):
    image = image.resize(IMG_SIZE)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = preprocess_fn(image_array)
    return np.expand_dims(image_array, axis=0)

# Prediction
def ensemble_predict(image):
    input1 = preprocess_image_for_model(image, efficientnet_preprocess)
    input2 = preprocess_image_for_model(image, mobilenet_preprocess)
    input3 = preprocess_image_for_model(image, densenet_preprocess)

    preds1 = model1.predict(input1, verbose=0)[0]
    preds2 = model2.predict(input2, verbose=0)[0]
    preds3 = model3.predict(input3, verbose=0)[0]
    
    # Weighted average
    ensemble_pred = (0.03 * preds1 + 0.53 * preds2 + 0.44 * preds3)
    
    predicted_class = np.argmax(ensemble_pred)
    confidence = ensemble_pred[predicted_class]
    return CLASS_LABELS[predicted_class], confidence, ensemble_pred

# UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image and the ensemble model will predict the disease class.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    
    with st.spinner("🔍 Predicting..."):
        predicted_label, confidence, raw_probs = ensemble_predict(image)

    st.success(f"✅ **Predicted Disease**: `{predicted_label}`")
    st.write(f"📈 **Confidence**: `{confidence * 100:.2f}%`")

    # 📊 Show class probabilities
    st.subheader("📊 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": CLASS_LABELS,
        "Probability": raw_probs
    }).set_index("Class")
    st.bar_chart(prob_df)
