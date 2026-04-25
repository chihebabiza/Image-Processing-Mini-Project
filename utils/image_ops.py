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


def canny_edge(img):
    gray = to_gray(img)
    return cv2.Canny(gray, 100, 200)

def sobel_edge(img):
    gray = to_gray(img)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    edges = np.sqrt(gx**2 + gy**2)
    edges = np.uint8(np.clip(edges, 0, 255))

    return edges

def prewitt_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)

    edges = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(edges, 0, 255))

def laplacian_edge(img):
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap), 0, 255))
    return lap

def roberts_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0],
                        [0, -1]])

    kernely = np.array([[0, 1],
                        [-1, 0]])

    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)

    edges = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(edges, 0, 255))


def image_to_bytes(img):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def log_edge(img, ksize=5):
    gray = to_gray(img)

    # Step 1: Gaussian blur (noise reduction)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Step 2: Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    log = np.uint8(np.clip(np.abs(log), 0, 255))
    return log