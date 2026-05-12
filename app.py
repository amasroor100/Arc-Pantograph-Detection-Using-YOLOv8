import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("⚡ Arc / Spark Detection System")

# Load model
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Prediction
        results = model.predict(tmp.name)

        # Show result
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)