import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Vasculitis vs Vascular Tumors Classifier", page_icon="🩺")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vasculitis_vs_vascular_tumors.keras")

model = load_model()
class_names = ["Vascular Tumors", "Vasculitis"]

st.title("Vasculitis vs Vascular Tumors — Image Classifier")
st.write(
    "Upload a dermatology image and this app will predict whether it shows "
    "**Vascular Tumors** or **Vasculitis**, using a MobileNetV2 transfer-learning model."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_resized = image.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")