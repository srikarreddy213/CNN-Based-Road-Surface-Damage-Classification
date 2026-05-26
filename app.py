import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import gdown

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide"
)

# =========================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================

MODEL_PATH = "road_damage_model.keras"

if not os.path.exists(MODEL_PATH):

    file_id = "1bVyGak95iSWEpOxJxBL7hksgozoGaSSx"

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with st.spinner("Downloading CNN Model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================================
# LOAD LABEL MAP
# =========================================

with open("label_map.json", "r") as f:
    label_map = json.load(f)

index_to_label = {v: k for k, v in label_map.items()}

# =========================================
# FIXED IMAGE SIZE
# =========================================

IMG_SIZE = 224

# =========================================
# SIDEBAR SETTINGS
# =========================================

st.sidebar.title("⚙️ Prediction Settings")

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
    "Rotation",
    -45,
    45,
    0
)

flip_option = st.sidebar.selectbox(
    "Flip Image",
    ["None", "Horizontal", "Vertical"]
)

show_probabilities = st.sidebar.checkbox(
    "Show Class Probabilities",
    value=True
)

# =========================================
# TITLE
# =========================================

st.title("🛣️ RoadGuard AI")
st.subheader("CNN-Based Road Damage Detection System")

st.markdown("""
Upload a road image to detect:

- Crack
- Pothole
- Patch
- Normal Road

using Deep Learning and CNN.
""")

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# IMAGE PREPROCESSING
# =========================================

def preprocess_image(image):

    # RESIZE
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # RGB FORMAT
    image = image.convert("RGB")

    # BRIGHTNESS
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    # CONTRAST
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # ROTATION
    image = image.rotate(rotation)

    # FLIP
    if flip_option == "Horizontal":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    elif flip_option == "Vertical":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # CONVERT TO ARRAY
    img_array = np.array(image)

    # NORMALIZE
    img_array = img_array.astype("float32") / 255.0

    # ADD BATCH DIMENSION
    img_array = np.expand_dims(img_array, axis=0)

    return image, img_array

# =========================================
# MAIN PREDICTION
# =========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.subheader("📷 Original Image")
        st.image(np.array(image), width=350)

    # PROCESS IMAGE
    processed_display_image, processed_image = preprocess_image(image)

    # PROCESSED IMAGE
    with col2:
        st.subheader("⚡ Processed Image")
        st.image(np.array(processed_display_image), width=350)

    # DEBUG SHAPE
    st.write("Prediction Input Shape:", processed_image.shape)

    # PREDICT BUTTON
    if st.button("🔍 Predict Road Condition"):

        with st.spinner("Analyzing Road Image..."):

            prediction = model.predict(processed_image)

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction)

            predicted_label = index_to_label[predicted_class]

        # =========================================
        # RESULTS
        # =========================================

        st.success(f"✅ Predicted Class: {predicted_label}")

        st.info(f"🎯 Confidence Score: {confidence*100:.2f}%")

        # =========================================
        # ALERTS
        # =========================================

        if predicted_label.lower() == "pothole":

            st.error("⚠️ Severe Road Damage Detected")

        elif predicted_label.lower() == "crack":

            st.warning("⚠️ Crack Detected on Road Surface")

        elif predicted_label.lower() == "patch":

            st.warning("⚠️ Road Patch Detected")

        else:

            st.success("✅ Road Appears Normal")

        # =========================================
        # PROBABILITY CHART
        # =========================================

        if show_probabilities:

            st.subheader("📊 Prediction Probabilities")

            probabilities = {}

            for i, prob in enumerate(prediction[0]):

                probabilities[index_to_label[i]] = float(prob)

            st.bar_chart(probabilities)

            for label, prob in probabilities.items():

                st.write(f"{label}: {prob*100:.2f}%")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.write("🚀 Built using TensorFlow + Streamlit")
