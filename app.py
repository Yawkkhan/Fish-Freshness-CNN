
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FishVision AI",
    page_icon="🐟",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "fish_freshness_final.keras"
)

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "results",
    "confusion_matrix.png"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #e0f2fe, #f0f9ff);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #475569;
}

.card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    background: white;
    margin-bottom: 20px;
}

.result-fresh {
    padding: 25px;
    border-radius: 15px;
    background: #dcfce7;
    border: 1px solid #86efac;
    color: #166534;
}

.result-spoiled {
    padding: 25px;
    border-radius: 15px;
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #991b1b;
}

.confidence-container {
    margin-top: 15px;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🐟 FishVision AI</h1>

<p>
AI-powered fish freshness detection using a Convolutional Neural Network.
Upload a fish image and let the model predict whether it is
<strong>Fresh</strong> or <strong>Spoiled</strong>.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>🧠 Model</h3>
    <p>Custom Convolutional Neural Network</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>📐 Input</h3>
    <p>128 × 128 × 3 RGB Image</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>🎯 Output</h3>
    <p>Binary Classification using Sigmoid</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("## 📤 Upload Fish Image")

uploaded_file = st.file_uploader(
    "Choose a fish image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    with col1:

        st.markdown("""
        <div class="card">
        <h3>🖼️ Uploaded Image</h3>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            caption="Uploaded Fish Image",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.markdown("""
        <div class="card">
        <h3>🤖 AI Prediction</h3>
        """, unsafe_allow_html=True)

        # Resize image
        img = image.resize((128, 128))

        # Convert to NumPy
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = float(model.predict(img_array, verbose=0)[0][0])

        # Classification
        if prediction >= 0.5:

            result = "SPOILED"
            confidence = prediction * 100

            st.markdown(f"""
            <div class="result-spoiled">
                <h2>⚠️ SPOILED</h2>
                <p>The model predicts that the fish is spoiled.</p>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            result = "FRESH"
            confidence = (1 - prediction) * 100

            st.markdown(f"""
            <div class="result-fresh">
                <h2>✅ FRESH</h2>
                <p>The model predicts that the fish is fresh.</p>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("### Confidence")

        st.progress(
            int(min(confidence, 100))
        )


# ============================================================
# HOW THE AI WORKS
# ============================================================

st.markdown("---")

st.markdown("## 🔬 How the AI Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>1️⃣ Upload</h3>
    <p>
    The user uploads an image of a fish.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>2️⃣ CNN Analysis</h3>
    <p>
    The CNN extracts visual features from the image.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>3️⃣ Prediction</h3>
    <p>
    The model classifies the fish as Fresh or Spoiled.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.markdown("---")

st.markdown("## 🧠 Model Architecture")

st.code("""
Input Image
    ↓
128 × 128 × 3
    ↓
Convolutional Layers
    ↓
Batch Normalization
    ↓
Max Pooling
    ↓
Dropout
    ↓
Global Average Pooling
    ↓
Dense Layer
    ↓
Sigmoid Output
    ↓
Fresh / Spoiled
""", language="text")


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.markdown("---")

st.markdown("## ⚙️ Technical Details")

st.write("""
**Architecture:** Custom CNN  
**Input Size:** 128 × 128 × 3  
**Classes:** Fresh / Spoiled  
**Output Activation:** Sigmoid  
**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Training Epochs:** 30  
""")


# ============================================================
# CONFUSION MATRIX
# ============================================================

if os.path.exists(CONFUSION_MATRIX_PATH):

    st.markdown("---")

    st.markdown("## 📊 Model Evaluation")

    st.image(
        CONFUSION_MATRIX_PATH,
        caption="Confusion Matrix",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#64748b;">

<p>
🐟 <strong>FishVision AI</strong>
</p>

<p>
Fish Freshness Detection using Convolutional Neural Network
</p>

<p class="small-text">
Academic Project
</p>

</div>
""", unsafe_allow_html=True)
