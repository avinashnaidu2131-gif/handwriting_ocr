import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

st.title("AI Handwritten OCR Demo")

uploaded_file = st.file_uploader("Upload handwritten image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simple preprocessing
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(thresh, detail=0)

    text_output = " ".join(result)

    st.subheader("Extracted Text")
    st.write(text_output)