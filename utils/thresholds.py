import cv2
from utils.image_ops import to_gray

def apply_otsu_threshold(img):
    gray = to_gray(img)
    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return th

def apply_threshold(img, thresh):
    gray = to_gray(img)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return th

def apply_double_threshold(img, low_thresh, high_thresh):
    gray = to_gray(img)
    
    mask = cv2.inRange(gray, low_thresh, high_thresh)
    
    return mask
