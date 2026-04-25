import streamlit as st
from utils.filters import apply_filter

def apply_filter_ui(image):
    filter_type = st.sidebar.selectbox(
        "Filter",
        ["mean", "gaussian", "median"]
    )

    ksize = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)

    if filter_type == "gaussian":
        sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)
        return apply_filter(image, filter_type, ksize, sigma=sigma)

    return apply_filter(image, filter_type, ksize)