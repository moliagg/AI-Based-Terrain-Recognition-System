import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model and define class names
model = load_model("terrain_classifier.h5")
class_names = ['Desert', 'Forest', 'Mountain', 'Plains']

# Preprocess function
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit App UI
st.set_page_config(page_title="Geospatial Terrain Recognition", layout="centered")
st.sidebar.title("🌍 Minor Project - Geospatial Recognition System")

st.title("🧭 Geospatial Terrain Recognition")
st.markdown("#### 📌 Please upload only **terrain images** like forests, mountains, plains, or deserts.")

uploaded_file = st.file_uploader("Upload a terrain image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Confirm it's a terrain image
    if st.checkbox("✅ I confirm this is a terrain image (not animals, flowers, people, etc.)"):
        st.write("Classifying terrain...")

        # Process and Predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = float(np.max(prediction))
        predicted_class = class_names[np.argmax(prediction)]

        # Confidence Bar
        st.progress(int(confidence * 100))  # Ensure it's int

        # Show result
        if confidence < 0.8:
            st.error(f"⚠️ Low confidence ({confidence * 100:.2f}%). The model is unsure about this terrain.")
        else:
            st.subheader(f"🧭 Predicted Terrain: **{predicted_class}**")
            st.success(f"✅ Confidence: {confidence * 100:.2f}%")
    else:
        st.info("☝️ Please confirm this is a valid terrain image to proceed.")

