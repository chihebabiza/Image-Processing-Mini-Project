import cv2
import numpy as np
from utils.image_ops import to_gray

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

def log_edge(img, ksize=5):
    gray = to_gray(img)

    # Step 1: Gaussian blur (noise reduction)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Step 2: Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    log = np.uint8(np.clip(np.abs(log), 0, 255))
    return log
