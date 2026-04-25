import streamlit as st
import cv2
from utils.image_ops import to_gray


def show_histograms(image, result):

    gray_original = to_gray(image)

    gray_processed = to_gray(result) if len(result.shape) == 3 else result

    hist_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    hist_processed = cv2.calcHist([gray_processed], [0], None, [256], [0, 256])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Histogram")
        st.line_chart(hist_original)

    with col2:
        st.subheader("Processed Histogram")
        st.line_chart(hist_processed)