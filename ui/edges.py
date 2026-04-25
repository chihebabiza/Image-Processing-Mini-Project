import streamlit as st
from utils.edges import canny_edge,sobel_edge,prewitt_edge,roberts_edge,laplacian_edge,log_edge

def apply_edge_ui(image):
    method = st.sidebar.selectbox(
        "Edge Method",
        [
            "Canny",
            "Sobel",
            "Prewitt",
            "Roberts",
            "Laplacian",
            "Laplacian of Gaussian (LoG)"
        ]
    )

    if method == "Canny":
        return canny_edge(image)

    elif method == "Sobel":
        return sobel_edge(image)

    elif method == "Prewitt":
        return prewitt_edge(image)

    elif method == "Roberts":
        return roberts_edge(image)

    elif method == "Laplacian":
        return laplacian_edge(image)

    # LoG
    ksize = st.sidebar.slider("Gaussian Kernel Size", 3, 11, 5, step=2)
    return log_edge(image, ksize)