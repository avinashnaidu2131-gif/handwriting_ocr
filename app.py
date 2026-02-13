import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.title("Hybrid AI Handwriting OCR")

uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png","jpg","jpeg"]
)

# ---------- Load Models ----------
@st.cache_resource
def load_easy():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    return processor, model

easy_reader = load_easy()
processor, trocr_model = load_trocr()

# ---------- Run ----------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input", use_column_width=True)

    # Preprocessing
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.5,(8,8))
    enhanced = clahe.apply(gray)

    thresh = cv2.threshold(
        enhanced,0,255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )[1]

    # EasyOCR
    easy_text = easy_reader.readtext(
        thresh,
        detail=0
    )

    # TrOCR
    rgb = Image.fromarray(img).convert("RGB")
    pixel = processor(
        images=rgb,
        return_tensors="pt"
    ).pixel_values

    ids = trocr_model.generate(pixel)
    trocr_text = processor.batch_decode(
        ids,
        skip_special_tokens=True
    )[0]

    # Display
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("EasyOCR Output")
        st.write("\n".join(easy_text))

    with col2:
        st.subheader("TrOCR Output")
        st.write(trocr_text)
