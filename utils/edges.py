import cv2
import numpy as np
from utils.image_ops import to_gray

def canny_edge(img, low_threshold=100, high_threshold=200):
    gray = to_gray(img)

    # Reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.Canny(blurred, low_threshold, high_threshold)


def sobel_edge(img):
    gray = to_gray(img)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    edges = np.sqrt(gx**2 + gy**2)

    return cv2.convertScaleAbs(edges)


def prewitt_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernely)

    edges = np.sqrt(gx**2 + gy**2)

    return cv2.convertScaleAbs(edges)


def roberts_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0],
                        [0, -1]], dtype=np.float32)

    kernely = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernely)

    edges = np.sqrt(gx**2 + gy**2)

    return cv2.convertScaleAbs(edges)


def laplacian_edge(img):
    gray = to_gray(img)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    lap = cv2.Laplacian(blurred, cv2.CV_64F)

    return cv2.convertScaleAbs(np.abs(lap))


def log_edge(img, ksize=5, sigma=1.0):
    gray = to_gray(img)

    # Ensure odd kernel size
    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1

    # Gaussian smoothing
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

    # Laplacian step
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    return cv2.convertScaleAbs(np.abs(log))