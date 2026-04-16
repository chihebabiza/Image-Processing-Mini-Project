import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from utils.image_ops import (
    to_gray,
    apply_threshold,
    histogram_equalization,
    histogram_stretching,
    edge_detection,
    image_to_bytes,
)

from utils.filters import add_noise, apply_filter
from utils.segmentation import segmentation_kmeans

st.set_page_config(page_title="Image Processing App", layout="wide")

st.title("4th Year AI Engineering - Image Processing Mini Project")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.sidebar.title("Operations")

    operation = st.sidebar.selectbox(
        "Choose operation",
        [
            "None",
            "Grayscale",
            "Threshold",
            "Histogram Equalization",
            "Histogram Stretching",
            "Noise",
            "Filtering",
            "Edge Detection",
            "Segmentation",
        ],
    )

    result = image

    if operation == "Grayscale":
        result = to_gray(image)

    elif operation == "Threshold":
        thresh = st.sidebar.slider("Threshold", 0, 255, 127)
        result = apply_threshold(image, thresh)

    elif operation == "Histogram Equalization":
        result = histogram_equalization(image)

    elif operation == "Histogram Stretching":
        result = histogram_stretching(image)

    elif operation == "Noise":
        noise_type = st.sidebar.selectbox("Noise Type", ["gaussian", "salt_pepper"])
        result = add_noise(image, noise_type)

    elif operation == "Filtering":
        filter_type = st.sidebar.selectbox("Filter", ["blur", "median", "sharpen"])
        result = apply_filter(image, filter_type)

    elif operation == "Edge Detection":
        result = edge_detection(image)

    elif operation == "Segmentation":
        k = st.sidebar.slider("Clusters (K)", 2, 8, 3)
        result = segmentation_kmeans(image, k)
    with col2:
        st.subheader("Processed Image")
        st.image(result, use_container_width=True)

    st.sidebar.download_button(
        "Download Result",
        data=image_to_bytes(
            result if len(result.shape) == 3 else np.stack([result] * 3, axis=-1)
        ),
        file_name="processed.png",
        mime="image/png",
    )

    st.subheader("Histogram")
    gray = to_gray(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    st.line_chart(hist)

else:
    st.info("Upload an image to start processing")
