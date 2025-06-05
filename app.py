import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("penalty_classifier.keras", compile=False)


#model = tf.keras.models.load_model("penalty_classifier.keras")

# Define class labels
class_labels = ['Centre', 'Left Lower', 'Left Upper', 'Right Lower', 'Right Upper']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("⚽ Penalty Kick Classifier")
st.markdown("Upload an image to classify its direction:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"### 🧠 Prediction: `{predicted_class}`")
    st.write("### 📊 Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob:.2f}")