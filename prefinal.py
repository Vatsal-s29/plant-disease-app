import streamlit as st
import streamlit.components.v1 as components  # ✅ Added for mobile detection
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

st.markdown("# 🌱 Welcome to the Plant Disease Detection App!")
st.write("This app uses an ensemble of deep learning models to predict the presence of plant diseases.")
st.markdown("---")

# -----------------------
# Supported Species Dropdown
# -----------------------
st.markdown("## 🌾 Supported Species & Diseases")

# Static species and diseases mapping
static_species_diseases = [
    ("🍎 Apple", [
        "Apple scab",
        "Black rot",
        "Cedar apple rust",
        "healthy"
    ]),
    ("🫐 Blueberry", [
        "healthy"
    ]),
    ("🍒 Cherry (including sour)", [
        "Powdery mildew",
        "healthy"
    ]),
    ("🌽 Corn (maize)", [
        "Cercospora leaf spot / Gray leaf spot",
        "Common rust",
        "Northern Leaf Blight",
        "healthy"
    ]),
    ("🍇 Grape", [
        "Black rot",
        "Esca (Black Measles)",
        "Leaf blight (Isariopsis Leaf Spot)",
        "healthy"
    ]),
    ("🍊 Orange", [
        "Haunglongbing (Citrus greening)"
    ]),
    ("🍑 Peach", [
        "Bacterial spot",
        "healthy"
    ]),
    ("🫑 Pepper, bell", [
        "Bacterial spot",
        "healthy"
    ])
]

# Color mapping
color_map = {
    "🍎": "#d62828",
    "🫐": "#4f518c",
    "🍒": "#b5179e",
    "🌽": "#f4a261",
    "🍇": "#6a0572",
    "🍊": "#f77f00",
    "🍑": "#ffb347",
    "🫑": "#2a9d8f"
}

# Two per row display (compact layout + colored list items)
for i in range(0, len(static_species_diseases), 2):
    cols = st.columns(2)
    for col_idx in range(2):
        if i + col_idx < len(static_species_diseases):
            species_name, disease_list = static_species_diseases[i + col_idx]
            emoji = species_name.split(" ")[0]
            color = color_map.get(emoji, "#000000")  # Fallback to black
            with cols[col_idx].expander(species_name):
                for disease in disease_list:
                    st.markdown(
                        f"<li style='margin-left:10px; font-size:15px; color:{color}'>{disease}</li>",
                        unsafe_allow_html=True
                    )

st.markdown("Upload an image of a plant leaf, and this ensemble-powered model will predict the disease class with high accuracy.")

# Check if mobile
is_mobile = st.query_params.get("mobile", ["false"])[0] == "true"

# Upload Box
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ✅ Reset session state for new image upload
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.top3_labels = None
        st.session_state.label_index = 0

    columns = st.columns(1 if is_mobile else 2)
    col1, col2 = columns
    with col1:
        st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)

    with col2:
        # Run prediction
        predicted_label, confidence, probabilities = ensemble_predict(image)

        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_labels = [CLASS_LABELS[idx] for idx in top3_indices]

        # Initialize session state variables if not already set
        if st.session_state.top3_labels is None:
            st.session_state.top3_labels = top3_labels

        # Get the current label to show
        current_label = st.session_state.top3_labels[st.session_state.label_index]

        # Display result
        st.success("✅ Prediction Complete!")
        st.markdown(f"### 🦠 Likely Disease: `{current_label}`")

        # ✅ Styled repredict button
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #fff3e0;
                color: #f4a261;
                padding: 0.25rem 0.75rem;
                font-size: 14px;
                border-radius: 5px;
                border: 1px solid #f4a261;
                transition: background-color 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background-color: #f4a261;
                color: white;
                border-color: #f4a261;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("Wrong Prediction? → Repredict"):
            st.session_state.label_index = (st.session_state.label_index + 1) % len(st.session_state.top3_labels)
            st.rerun()

    # 💡 AI Assistance Chatbot
    st.markdown(
        """
        <a href="https://549e1cfa6993f02828.gradio.live" target="_blank" style="text-decoration: none;">
            <div style="margin-top: 10px; margin-bottom: 10px; padding: 10px; border-left: 5px solid #4a90e2; background-color: #e6f0ff; border-radius: 5px; color: black;">
                💡 <strong>Get AI assisted help</strong>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📊 Show All Class Probabilities", expanded=False):
        prob_df = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Probability": probabilities
        }).set_index("Class").sort_values("Probability", ascending=True)
        st.bar_chart(prob_df)

else:
    # 💡 AI Assistance Chatbot
    st.markdown(
        """
        <a href="https://ecf82da8b217e9232e.gradio.live" target="_blank" style="text-decoration: none;">
            <div style="margin-top: 10px; padding: 10px; border-left: 5px solid #4a90e2; background-color: #e6f0ff; border-radius: 5px; color: black;">
                💡 <strong>Get AI assisted help</strong>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
