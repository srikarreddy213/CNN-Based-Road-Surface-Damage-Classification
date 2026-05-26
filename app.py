# 🛣️ Interactive Road Damage Detection Streamlit App

# =========================================
# app.py
# INTERACTIVE ROAD DAMAGE DETECTION SYSTEM
# =========================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import json
import cv2

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource

def load_model():
    model = tf.keras.models.load_model("road_damage_model.h5")
    return model

model = load_model()

# -----------------------------------------
# LOAD LABEL MAP
# -----------------------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)

index_to_label = {v:k for k,v in label_map.items()}

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.title("⚙️ Prediction Settings")

img_size = st.sidebar.selectbox(
    "Select Image Size",
    [128, 224, 256],
    index=1
)

brightness = st.sidebar.slider(
    "Brightness",
    0.5,
    2.0,
    1.0,
    0.1
)

contrast = st.sidebar.slider(
    "Contrast",
    0.5,
    2.0,
    1.0,
    0.1
)

rotation = st.sidebar.slider(
    "Rotation Angle",
    -45,
    45,
    0
)

flip_option = st.sidebar.selectbox(
    "Flip Image",
    ["None", "Horizontal", "Vertical"]
)

show_probabilities = st.sidebar.checkbox(
    "Show All Class Probabilities",
    value=True
)

# -----------------------------------------
# TITLE
# -----------------------------------------
st.title("🛣️ AI Road Damage Detection System")

st.markdown("""
This application predicts road damage categories using a CNN model.

### Features
- Upload road images
- Adjust brightness
- Adjust contrast
- Rotate images
- Flip images
- Get prediction confidence
- Interactive CNN prediction
""")

# -----------------------------------------
# FILE UPLOAD
# -----------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------
# IMAGE PROCESSING FUNCTION
# -----------------------------------------

def preprocess_image(image):

    # Resize
    image = image.resize((img_size, img_size))

    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Rotation
    image = image.rotate(rotation)

    # Flip
    if flip_option == "Horizontal":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    elif flip_option == "Vertical":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    return image, img_array

# -----------------------------------------
# PREDICTION SECTION
# -----------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    # PROCESS IMAGE
    processed_display_image, processed_image = preprocess_image(image)

    # PROCESSED IMAGE
    with col2:
        st.subheader("⚡ Processed Image")
        st.image(processed_display_image, use_container_width=True)

    # PREDICT BUTTON
    if st.button("🔍 Predict Road Condition"):

        with st.spinner("Model Predicting..."):

            prediction = model.predict(processed_image)

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction)

            predicted_label = index_to_label[predicted_class]

        # -----------------------------------------
        # RESULTS
        # -----------------------------------------
        st.success(f"✅ Predicted Class: {predicted_label}")

        st.info(f"🎯 Confidence Score: {confidence*100:.2f}%")

        # -----------------------------------------
        # DAMAGE ALERTS
        # -----------------------------------------
        if predicted_label.lower() == "pothole":
            st.error("⚠️ Severe Road Damage Detected")

        elif predicted_label.lower() == "crack":
            st.warning("⚠️ Crack Detected on Road Surface")

        elif predicted_label.lower() == "patch":
            st.warning("⚠️ Road Patch Detected")

        else:
            st.success("✅ Road Appears Normal")

        # -----------------------------------------
        # PROBABILITY CHART
        # -----------------------------------------
        if show_probabilities:

            st.subheader("📊 Prediction Probabilities")

            probabilities = {}

            for i, prob in enumerate(prediction[0]):
                probabilities[index_to_label[i]] = float(prob)

            st.bar_chart(probabilities)

            # SHOW VALUES
            for label, prob in probabilities.items():
                st.write(f"{label} : {prob*100:.2f}%")

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")

st.write("🚀 Built Using TensorFlow + Streamlit")
