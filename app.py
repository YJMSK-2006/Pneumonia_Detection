import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model - update the path if saved differently
model = tf.keras.models.load_model("model.h5")

st.title("Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert image to grayscale if needed
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    # Preprocess the image to model input shape - adjust size to your model's requirement
    img = image.resize((150, 150))  # example: change (150,150) to your input size
    img_array = np.array(img) / 255.0  # normalize if required by model
    img_array = img_array.reshape(1, 150, 150, 1)  # add batch and channels dimension if model expects it

    if st.button("Predict"):
        prediction = model.predict(img_array)
        # Assuming binary classifier with sigmoid activation output
        if prediction[0][0] > 0.5:
            st.error("Prediction: Pneumonia Detected")
        else:
            st.success("Prediction: Normal")
