import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from class_names import class_names

# Load model
model = load_model("models/skin_disease_model.keras")

st.title("AI Skin Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    THRESHOLD = 70

    if confidence < THRESHOLD:
        st.warning(
            f"""
            Low confidence prediction ({confidence:.2f}%)

            This image may not belong to any disease class present in the training dataset.
            """
        )

    st.success(f"Disease: {class_names[predicted_class]}")
    st.info(f"Confidence: {confidence:.2f}%")