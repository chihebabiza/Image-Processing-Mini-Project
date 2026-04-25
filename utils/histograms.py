import cv2
import numpy as np
from utils.image_ops import to_gray


def histogram_equalization(img):
    gray = to_gray(img)
    return cv2.equalizeHist(gray)


def histogram_stretching(img):
    gray = to_gray(img)
    min_val, max_val = np.min(gray), np.max(gray)

    if max_val == min_val:
        return gray

    stretched = (gray - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)