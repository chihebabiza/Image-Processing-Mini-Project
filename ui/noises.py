import streamlit as st
from utils.noises import add_noise

def apply_noise_ui(image):
    noise_type = st.sidebar.selectbox(
        "Noise Type",
        ["Gaussian", "Salt Pepper"]
    )

    if noise_type == "Gaussian":
        std = st.sidebar.slider("Sigma", 1, 100, 25)
        return add_noise(image, "gaussian", std=std)

    prob = st.sidebar.slider("Probability", 0.0, 0.2, 0.02)
    return add_noise(image, "salt_pepper", prob=prob)