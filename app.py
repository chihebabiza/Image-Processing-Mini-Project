import streamlit as st
import numpy as np
from PIL import Image

from utils.histograms import histogram_equalization, histogram_stretching
from ui.noises import apply_noise_ui
from ui.filters import apply_filter_ui
from ui.edges import apply_edge_ui
from ui.thresholds import apply_threshold_ui
from ui.histograms import show_histograms
from utils.image_ops import image_to_bytes,to_gray


st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("Image Processing App")


@st.cache_data
def load_image(file):
    return np.array(Image.open(file))


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg","tif","tiff"])

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
        ],
    )

    result = image

    if operation == "Grayscale":
        result = to_gray(image)

    elif operation == "Threshold":
        result = apply_threshold_ui(image)

    elif operation == "Histogram Equalization":
        result = histogram_equalization(image)

    elif operation == "Histogram Stretching":
        result = histogram_stretching(image)

    elif operation == "Edge Detection":
        result = apply_edge_ui(image)

    elif operation == "Noise":
        result = apply_noise_ui(image)

    elif operation == "Filtering":
        result = apply_filter_ui(image)

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