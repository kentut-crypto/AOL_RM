import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Load Keras model
model = tf.keras.models.load_model("resnet_model.keras")
class_names = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Preprocess image
def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction and return class + confidence
def predict_image(img, model):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    return predicted_index, confidence, predictions

# Streamlit UI
st.set_page_config(page_title="Corn Leaf Disease Detector", layout="centered")
st.title("🌽 Corn Leaf Disease Detection")
st.write("Upload an image of a corn leaf to detect possible diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            pred_class, confidence, raw_probs = predict_image(uploaded_file, model)
            st.success("Prediction complete!")
            st.markdown(f"**Predicted Disease:** `{class_names[pred_class]}`")
            st.markdown(f"**Confidence:** `{confidence:.2%}`")

            st.subheader("Class Probabilities")
            prob_dict = {class_names[i]: f"{prob:.2%}" for i, prob in enumerate(raw_probs)}
            st.json(prob_dict)