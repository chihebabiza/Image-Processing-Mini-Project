import cv2
import numpy as np
from PIL import Image
import io


def to_gray(img):
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return img


def apply_threshold(img, thresh):
    gray = to_gray(img)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return th

def apply_double_threshold(img, low_thresh, high_thresh):
    gray = to_gray(img)
    
    mask = cv2.inRange(gray, low_thresh, high_thresh)
    
    return mask


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


def edge_detection(img):
    gray = to_gray(img)
    return cv2.Canny(gray, 100, 200)


def image_to_bytes(img):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()