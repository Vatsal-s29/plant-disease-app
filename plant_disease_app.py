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
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

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

# Preprocessing
efficientnet_preprocess = efficientnet.preprocess_input
mobilenet_preprocess = mobilenet_v2.preprocess_input
densenet_preprocess = densenet.preprocess_input

@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("efficientnetb0_final.h5")
    model2 = tf.keras.models.load_model("mobilenetv2_finetuned.h5")
    model3 = tf.keras.models.load_model("densenet121_finetuned.h5")
    return model1, model2, model3

model1, model2, model3 = load_models()

def preprocess_image_for_model(image, preprocess_fn):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = preprocess_fn(img_array)
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(image):
    input1 = preprocess_image_for_model(image, efficientnet_preprocess)
    input2 = preprocess_image_for_model(image, mobilenet_preprocess)
    input3 = preprocess_image_for_model(image, densenet_preprocess)

    preds1 = model1.predict(input1, verbose=0)[0]
    preds2 = model2.predict(input2, verbose=0)[0]
    preds3 = model3.predict(input3, verbose=0)[0]

    # Weighted average
    final_pred = (0.03 * preds1 + 0.53 * preds2 + 0.44 * preds3)

    predicted_class = np.argmax(final_pred)
    confidence = final_pred[predicted_class]
    return CLASS_LABELS[predicted_class], confidence, final_pred

# UI
st.markdown("## 🌱 Welcome to the Plant Disease Detection App!")
st.markdown("Upload an image of a plant leaf, and this ensemble-powered model will predict the disease class with high accuracy.")
st.markdown("---")

# Centered Upload Box
col_center = st.columns([2, 4, 2])
with col_center[1]:
    uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("🔍 Running ensemble prediction..."):
            predicted_label, confidence, probabilities = ensemble_predict(image)
        
        st.success("✅ Prediction Complete!")
        st.markdown(f"### 🦠 Disease: `{predicted_label}`")

    with st.expander("📊 Show All Class Probabilities", expanded=False):
        prob_df = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Probability": probabilities
        }).set_index("Class").sort_values("Probability", ascending=True)
        st.bar_chart(prob_df)

# -----------------------
# Supported Species Dropdown
# -----------------------
st.markdown("## 🌾 Supported Species & Diseases")

# Extract species and group diseases
from collections import defaultdict
species_diseases = defaultdict(list)

for entry in CLASS_LABELS:
    if "___" in entry:
        species, disease = entry.split("___")
    else:
        species, disease = entry, "healthy"
    species_diseases[species].append(disease)

for species, diseases in sorted(species_diseases.items()):
    with st.expander(f"🌿 {species.replace('_', ' ')}"):
        for disease in diseases:
            st.markdown(f"- {disease.replace('_', ' ')}")
