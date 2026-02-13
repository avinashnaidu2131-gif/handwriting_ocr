import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.title("Advanced Handwriting OCR (TrOCR)")

uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png","jpg","jpeg"]
)

# Load model once
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    return processor, model

processor, model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # Convert image for model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate text
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    st.subheader("Extracted Text")
    st.write(text)
