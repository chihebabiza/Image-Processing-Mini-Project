import streamlit as st
from utils.thresholds import apply_otsu_threshold, apply_threshold, apply_double_threshold

def apply_threshold_ui(image):
    mode = st.sidebar.selectbox(
        "Threshold Mode",
        ["Single Threshold", "Double Threshold", "Otsu"]
    )

    if mode == "Single Threshold":
        t = st.sidebar.slider("Threshold", 0, 255, 127)
        return apply_threshold(image, t)

    elif mode == "Double Threshold":
        low = st.sidebar.slider("Low Threshold", 0, 255, 50)
        high = st.sidebar.slider("High Threshold", 0, 255, 150)

        if low > high:
            low, high = high, low
            st.sidebar.warning("Swapped values to keep low ≤ high")

        return apply_double_threshold(image, low, high)

    # Otsu
    st.sidebar.info("Otsu automatically selects the best threshold")
    return apply_otsu_threshold(image)