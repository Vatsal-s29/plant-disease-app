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
import os
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

import google.generativeai as genai

# ✅ SET PAGE CONFIG FIRST - ONLY ONCE!
st.set_page_config(
    page_title="🌿 Plant Disease Classifier", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ✅ Configure API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("⚠️ Gemini API key not configured. Please set GEMINI_API_KEY in secrets.")

# ✅ Mobile detection function
def is_mobile_device():
    """Detect if user is on mobile using user agent."""
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        mobile_keywords = ['android', 'iphone', 'ipad', 'mobile', 'windows phone']
        return any(keyword in user_agent for keyword in mobile_keywords)
    except:
        return False

# ✅ Gemini analysis function
def get_gemini_analysis(image, predicted_label):
    """Get AI analysis of plant disease with error handling."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Convert PIL image → bytes
        import io
        img_byte_arr = io.BytesIO()
        image.convert("RGB").save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()
        
        prompt = f"""
You are an expert agricultural plant pathologist.

The predicted disease is: **{predicted_label}**

Based on this disease, analyze the uploaded leaf image and provide a concise report.

Format your response EXACTLY as shown below:

#### Severity: [1-5]/5
[One sentence explaining why this severity score]

#### Remedies

##### Chemical Remedies
1. **[Product Name]** - [One sentence about usage/effectiveness]
2. **[Product Name]** - [One sentence about usage/effectiveness]
3. **[Product Name]** - [One sentence about usage/effectiveness]

##### Natural Remedies
1. **[Method/Ingredient]** - [One sentence about application]
2. **[Method/Ingredient]** - [One sentence about application]
3. **[Method/Ingredient]** - [One sentence about application]

##### Best Farming Practices
- [One concise practice]
- [One concise practice]
- [One concise practice]
"""
        
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ]
        )
        
        return response.text if hasattr(response, "text") else str(response)
    
    except Exception as e:
        return f"⚠️ **Analysis Error**: {str(e)}\n\nPlease try again or check your API configuration."

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

@st.cache_resource
def load_models():
    """Load all three models with error handling."""
    try:
        model1 = tf.keras.models.load_model("efficientnetb0_final.h5")
        model2 = tf.keras.models.load_model("mobilenetv2_finetuned.h5")
        model3 = tf.keras.models.load_model("densenet121_finetuned.h5")
        return model1, model2, model3
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Load models
with st.spinner("🔄 Loading AI models..."):
    model1, model2, model3 = load_models()

def preprocess_image_for_model(image, preprocess_fn):
    """Preprocess image for specific model."""
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = preprocess_fn(img_array)
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(image):
    """Run ensemble prediction across all three models."""
    try:
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
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        return None, None, None

# ============================================
# UI SECTION
# ============================================

st.markdown("# 🌱 My Plant Buddy 🌱")
st.write("This app uses an ensemble of deep learning models to predict the presence of plant diseases.")
st.markdown("---")

# Supported Species Section
st.markdown("## 🌾 Supported Species & Diseases")

static_species_diseases = [
    ("🍎 Apple", ["Apple scab", "Black rot", "Cedar apple rust", "healthy"]),
    ("🫐 Blueberry", ["healthy"]),
    ("🍒 Cherry (including sour)", ["Powdery mildew", "healthy"]),
    ("🌽 Corn (maize)", ["Cercospora leaf spot / Gray leaf spot", "Common rust", "Northern Leaf Blight", "healthy"]),
    ("🍇 Grape", ["Black rot", "Esca (Black Measles)", "Leaf blight (Isariopsis Leaf Spot)", "healthy"]),
    ("🍊 Orange", ["Haunglongbing (Citrus greening)"]),
    ("🍑 Peach", ["Bacterial spot", "healthy"]),
    ("🫑 Pepper, bell", ["Bacterial spot", "healthy"])
]

color_map = {
    "🍎": "#d62828", "🫐": "#4f518c", "🍒": "#b5179e", "🌽": "#f4a261",
    "🍇": "#6a0572", "🍊": "#f77f00", "🍑": "#ffb347", "🫑": "#2a9d8f"
}

# Display species in two columns
for i in range(0, len(static_species_diseases), 2):
    cols = st.columns(2)
    for col_idx in range(2):
        if i + col_idx < len(static_species_diseases):
            species_name, disease_list = static_species_diseases[i + col_idx]
            emoji = species_name.split(" ")[0]
            color = color_map.get(emoji, "#000000")
            with cols[col_idx].expander(species_name):
                for disease in disease_list:
                    st.markdown(
                        f"<li style='margin-left:10px; font-size:15px; color:{color}'>{disease}</li>",
                        unsafe_allow_html=True
                    )

st.markdown("Upload an image of a plant leaf, and this ensemble-powered model will predict the disease class with high accuracy.")

# ✅ Detect if mobile
is_mobile = is_mobile_device()

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# ============================================
# PREDICTION SECTION
# ============================================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ✅ Initialize session state for new upload
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.top3_labels = None
        st.session_state.label_index = 0
        st.session_state.gemini_analyses = {}  # Cache analyses by label

    # Create layout based on device
    if is_mobile:
        st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)
        st.markdown("---")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)

    # Run prediction only once
    if st.session_state.top3_labels is None:
        with st.spinner("🔍 Analyzing image..."):
            predicted_label, confidence, probabilities = ensemble_predict(image)
            
            if predicted_label is None:
                st.error("Failed to make prediction. Please try again.")
                st.stop()
            
            # Get top 3 predictions
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            st.session_state.top3_labels = [CLASS_LABELS[idx] for idx in top3_indices]
            st.session_state.probabilities = probabilities

    # Get current label
    current_label = st.session_state.top3_labels[st.session_state.label_index]

    # Display section (in col2 if desktop, or below image if mobile)
    display_container = col2 if not is_mobile else st.container()
    
    with display_container:
        # ✅ Get or generate Gemini analysis for current label
        if current_label not in st.session_state.gemini_analyses:
            with st.spinner("🔍 Analyzing severity & treatment recommendations..."):
                st.session_state.gemini_analyses[current_label] = get_gemini_analysis(image, current_label)
        
        gemini_output = st.session_state.gemini_analyses[current_label]

        # Display prediction
        st.success("✅ Prediction Complete!")
        st.markdown(f"### 🦠 Likely Disease: `{current_label}`")
        
        # Confidence score
        current_idx = CLASS_LABELS.index(current_label)
        current_confidence = st.session_state.probabilities[current_idx]
        st.metric("Confidence", f"{current_confidence*100:.1f}%")

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

        if st.button("🔄 Wrong Prediction? → Try Next Best Match"):
            st.session_state.label_index = (st.session_state.label_index + 1) % len(st.session_state.top3_labels)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🌡️ Disease Severity & Treatment")
    st.markdown(gemini_output)

    st.markdown("---")
    # AI Assistance Link
    # st.markdown(
    #     """
    #     <div style="margin-top: 10px; margin-bottom: 10px; padding: 15px; border-left: 5px solid #4a90e2; background-color: #e6f0ff; border-radius: 5px;">
    #         💡 <strong>Need more help?</strong><br>
    #         <span style="font-size: 14px;">Chat with our AI assistant for personalized farming advice.</span>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    # Show all probabilities
    with st.expander("📊 Show All Class Probabilities", expanded=False):
        prob_df = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Probability": st.session_state.probabilities
        }).set_index("Class").sort_values("Probability", ascending=False)
        st.bar_chart(prob_df)

else:
    # Welcome message when no file uploaded
    st.info("👆 Upload a plant leaf image to get started!")
    
    # st.markdown("---")
    # st.markdown(
    #     """
    #     <div style="margin-top: 10px; padding: 15px; border-left: 5px solid #4a90e2; background-color: #e6f0ff; border-radius: 5px;">
    #         💡 <strong>Need help getting started?</strong><br>
    #         <span style="font-size: 14px;">Chat with our AI assistant for farming advice and tips.</span>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )