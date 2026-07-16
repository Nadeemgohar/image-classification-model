# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier | Nadeem Gohar",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for a premium look
# ----------------------------
st.markdown("""
    <style>
        /* Overall background */
        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #161925 100%);
            color: #e8e8ec;
        }

        /* Hide default Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Hero title block */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .hero-subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            margin-top: 4px;
            margin-bottom: 24px;
        }

        /* Card container */
        .premium-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            margin-bottom: 20px;
        }

        /* Prediction badge */
        .prediction-badge {
            display: inline-block;
            padding: 10px 22px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #16a34a);
            color: white;
            font-weight: 700;
            font-size: 1.3rem;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 20px rgba(34, 197, 94, 0.35);
        }

        .confidence-text {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-top: 8px;
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(167, 139, 250, 0.4);
            border-radius: 14px;
            padding: 10px;
            background: rgba(167, 139, 250, 0.03);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12141c 0%, #1a1d2b 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] a {
            color: #a78bfa !important;
        }

        /* Class chips */
        .chip {
            display: inline-block;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 8px;
            background: rgba(125, 211, 252, 0.1);
            border: 1px solid rgba(125, 211, 252, 0.25);
            color: #7dd3fc;
            font-size: 0.82rem;
        }

        hr {
            border-color: rgba(255,255,255,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def get_model():
    return load_model("cifar10_cnn_model.keras")

model = get_model()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

class_emojis = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱',
    'deer': '🦌', 'dog': '🐶', 'frog': '🐸', 'horse': '🐴',
    'ship': '🚢', 'truck': '🚚'
}

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🧠 About the Project")
    st.info("""
**Created by:** Nadeem Gohar

This project classifies images into 10 CIFAR-10 categories
using a Convolutional Neural Network (CNN) built with
TensorFlow/Keras.
""")
    st.markdown("---")
    st.markdown("### 📚 Supported Classes")
    chips_html = "".join(
        f"<span class='chip'>{class_emojis[c]} {c}</span>" for c in classes
    )
    st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: CNN · Dataset: CIFAR-10 · Framework: TensorFlow/Keras")

# ----------------------------
# Hero header
# ----------------------------
st.markdown('<div class="hero-title">Image Classification Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload an image and let the CNN model identify it across 10 categories — '
    'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.</div>',
    unsafe_allow_html=True
)

# ----------------------------
# Main layout
# ----------------------------
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Prediction")

    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            img = image.resize((32, 32))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            predicted_idx = int(np.argmax(prediction))
            predicted_class = classes[predicted_idx]
            confidence = float(np.max(prediction)) * 100

        st.markdown(
            f'<span class="prediction-badge">{class_emojis[predicted_class]} '
            f'{predicted_class.upper()}</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="confidence-text">Confidence: <b>{confidence:.2f}%</b></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Class probability breakdown**")

        probs_df = pd.DataFrame({
            "Class": classes,
            "Probability": (prediction[0] * 100)
        }).sort_values("Probability", ascending=True)

        st.bar_chart(
            probs_df.set_index("Class"),
            horizontal=True,
            color="#a78bfa"
        )
    else:
        st.markdown(
            "<p style='color:#6b7280;'>Upload an image on the left to see the "
            "model's prediction and confidence breakdown here.</p>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "Built with ❤️ by <b>Nadeem Gohar</b> · Powered by TensorFlow & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
