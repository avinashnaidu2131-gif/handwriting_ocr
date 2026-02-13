import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AI Handwritten OCR", layout="wide")

st.title("AI Handwritten OCR Demo")

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png", "jpg", "jpeg"]
)

# Initialize OCR reader once
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ---------- Processing ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast = clahe.apply(gray)

    # Noise Removal
    denoise = cv2.fastNlMeansDenoising(
        contrast, None, 30, 7, 21
    )

    # Sharpening
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    sharp = cv2.filter2D(denoise, -1, kernel)

    # Thresholding
    thresh = cv2.threshold(
        sharp, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    with col2:
        st.subheader("Processed Image")
        st.image(thresh, use_column_width=True)

    # ---------- OCR ----------
    with st.spinner("Extracting text..."):
        result = reader.readtext(thresh, detail=0)
        text_output = "\n".join(result)

    st.subheader("Extracted Text")
    st.text_area("", text_output, height=200)

    # ---------- Download ----------
    if text_output.strip():
        buffer = BytesIO(text_output.encode())
        st.download_button(
            label="Download Text",
            data=buffer,
            file_name="ocr_output.txt",
            mime="text/plain"
        )
