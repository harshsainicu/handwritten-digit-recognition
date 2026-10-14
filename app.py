import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("A beginner-friendly Deep Learning demo using a Convolutional Neural Network (CNN) trained on MNIST.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn.keras")
    return model

st.info("Upload a 28×28 handwritten digit image (0–9). The model predicts which digit it sees.")
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("L").resize((28, 28))
    arr = np.array(image).astype("float32")
    # MNIST uses white digits on a black background; invert typical black-on-white uploads.
    if arr.mean() > 127:
        arr = 255 - arr
    arr = arr / 255.0
    st.image(image, caption="Uploaded image", width=160)
    prediction = load_model().predict(arr.reshape(1, 28, 28, 1), verbose=0)[0]
    digit = int(np.argmax(prediction))
    st.success(f"Predicted digit: {digit}")
    st.write(f"Model confidence: {prediction[digit] * 100:.1f}%")
else:
    st.caption("For the simplest demo, upload a clear image containing one large handwritten digit.")
