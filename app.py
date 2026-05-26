import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import gdown
import pandas as pd

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #ff1f1f;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# DOWNLOAD MODEL
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
# SETTINGS
# =========================================

IMG_SIZE = 128

# =========================================
# SIDEBAR
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

# =========================================
# SECTION 1 — HEADER
# =========================================

st.title("🛣️ AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")

# =========================================
# SECTION 2 — ABOUT PROJECT
# =========================================

st.header("📘 About the Project")

st.write("""
Road monitoring is extremely important for maintaining safe transportation infrastructure.
Damaged roads can cause accidents, traffic congestion, vehicle damage, and safety risks.

This project uses Convolutional Neural Networks (CNNs), a deep learning technique used in computer vision,
to automatically detect road surface damage from images.

CNN models can identify patterns such as:
- potholes
- cracks
- patches
- manholes

### Industry Applications
- Smart City Monitoring
- Automated Road Inspection
- Government Infrastructure Management
- Traffic Safety Systems
- Autonomous Vehicle Vision Systems
""")

st.markdown("---")

# =========================================
# SECTION 3 — UPLOAD AREA
# =========================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload Road Surface Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# PREPROCESS FUNCTION
# =========================================

def preprocess_image(image):

    image = image.resize((128, 128))

    image = image.convert("RGB")

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    image = image.rotate(rotation)

    if flip_option == "Horizontal":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    elif flip_option == "Vertical":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return image, img_array

# =========================================
# MAIN SECTION
# =========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    processed_display_image, processed_image = preprocess_image(image)

    # =========================================
    # SECTION 4 — IMAGE PREVIEW
    # =========================================

    st.header("🖼️ Uploaded Image Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(np.array(image), width=350)

    with col2:
        st.subheader("Processed Image")
        st.image(np.array(processed_display_image), width=350)

    st.markdown("---")

    # =========================================
    # PREDICT BUTTON
    # =========================================

    if st.button("🔍 Predict Road Damage"):

        with st.spinner("Analyzing Road Condition..."):

            prediction = model.predict(processed_image)

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction)

            predicted_label = index_to_label[predicted_class]

        # =========================================
        # SEVERITY
        # =========================================

        severity = "Low"

        if predicted_label.lower() == "pothole":
            severity = "High"

        elif predicted_label.lower() == "crack":
            severity = "Medium"

        elif predicted_label.lower() == "patch":
            severity = "Medium"

        else:
            severity = "Low"

        # =========================================
        # SECTION 5 — PREDICTION AREA
        # =========================================

        st.header("📌 Prediction Results")

        st.success(
            f"Prediction: {predicted_label}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        st.warning(
            f"Severity Level: {severity}"
        )

        st.markdown("---")

        # =========================================
        # SECTION 6 — VISUALIZATION
        # =========================================

        st.header("📊 Visualization Area")

        probabilities = {}

        for i, prob in enumerate(prediction[0]):

            probabilities[index_to_label[i]] = float(prob)

        chart_data = pd.DataFrame(
            probabilities.items(),
            columns=["Class", "Probability"]
        )

        st.bar_chart(
            chart_data.set_index("Class")
        )

        st.write("### Class Confidence Scores")

        for label, prob in probabilities.items():

            st.write(
                f"{label}: {prob*100:.2f}%"
            )

        st.markdown("---")

        # =========================================
        # SECTION 7 — RECOMMENDATIONS
        # =========================================

        st.header("🚨 Recommendations")

        if severity == "High":

            st.error("""
Immediate maintenance recommended.

High-risk road condition detected.

Possible vehicle damage and accident risk.
""")

        elif severity == "Medium":

            st.warning("""
Road repair recommended soon.

Moderate infrastructure damage detected.
""")

        else:

            st.success("""
Road condition appears safe.

Routine monitoring recommended.
""")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.write(
    "🚀 Built using TensorFlow, CNN, and Streamlit"
)
