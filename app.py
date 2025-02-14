import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('densenet121_finetuned1.keras')

model = load_model()

# Class Names
class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

# Title
st.title("🦷 Teeth Classification App")
st.write("Upload an image of a tooth condition, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    def preprocess_image(image_file):
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    img_array = preprocess_image(uploaded_file)

    # Add a button for prediction
    if st.button("🔍 Classify"):
        with st.spinner("Classifying..."):
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Convert to percentage
            
            # Display Results
            st.success(f"🦷 **Prediction:** {class_names[class_index]}")
            st.write(f"🎯 **Confidence:** {confidence:.2f}%")