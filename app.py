import streamlit as st
import numpy as np
from PIL import Image

from utils.image_ops import (
    to_gray,
    apply_threshold,
    apply_double_threshold,
    histogram_equalization,
    histogram_stretching,
    edge_detection,
    image_to_bytes,
)
from utils.utils import show_histograms
from utils.filters import add_noise, apply_filter
from utils.segmentation import segmentation_kmeans


st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("Image Processing Lab")


@st.cache_data
def load_image(file):
    return np.array(Image.open(file))


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = load_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    st.sidebar.title("Controls")

    operation = st.sidebar.selectbox(
        "Operation",
        [
            "None",
            "Grayscale",
            "Histogram Equalization",
            "Histogram Stretching",
            "Noise",
            "Filtering",
            "Threshold",
            "Edge Detection",
            "Segmentation",
        ],
    )

    result = image

    if operation == "Grayscale":
        result = to_gray(image)

    elif operation == "Threshold":

        mode = st.sidebar.selectbox(
            "Threshold Mode",
            ["Single Threshold", "Double Threshold"]
        )

        if mode == "Single Threshold":
            t = st.sidebar.slider("Threshold", 0, 255, 127)
            result = apply_threshold(image, t)

        else:  # Double Threshold
            low = st.sidebar.slider("Low Threshold", 0, 255, 50)
            high = st.sidebar.slider("High Threshold", 0, 255, 150)

            if low > high:
                low, high = high, low
                st.sidebar.warning("Swapped values to keep low ≤ high")

            result = apply_double_threshold(image, low, high)

    elif operation == "Histogram Equalization":
        result = histogram_equalization(image)

    elif operation == "Histogram Stretching":
        result = histogram_stretching(image)

    elif operation == "Edge Detection":
        result = edge_detection(image)

    elif operation == "Noise":
        noise_type = st.sidebar.selectbox("Noise Type", ["Gaussian", "Salt Pepper"])

        if noise_type == "Gaussian":
            std = st.sidebar.slider("Sigma", 1, 100, 25)
            result = add_noise(image, "gaussian", std=std)

        else:
            prob = st.sidebar.slider("Probability", 0.0, 0.2, 0.02)
            result = add_noise(image, "salt_pepper", prob=prob)

    elif operation == "Filtering":
        f = st.sidebar.selectbox("Filter", ["mean", "gaussian", "median"])
        k = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)

        if f == "gaussian":
            sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)
            result = apply_filter(image, f, k, sigma=sigma)
        else:
            result = apply_filter(image, f, k)

    elif operation == "Segmentation":
        k = st.sidebar.slider("Clusters", 2, 8, 3)
        result = segmentation_kmeans(image, k)

    with col2:
        st.subheader("Processed")
        st.image(result, use_container_width=True)

    st.sidebar.download_button(
        "Download",
        data=image_to_bytes(
            result if len(result.shape) == 3 else np.stack([result] * 3, axis=-1)
        ),
        file_name="processed.png",
        mime="image/png",
    )

    if st.checkbox("Show Histograms"):
        show_histograms(image, result)

else:
    st.info("Upload an image to start")