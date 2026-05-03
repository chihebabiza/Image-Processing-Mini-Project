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

    edges = cv2.magnitude(gx, gy)

    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    return edges.astype(np.uint8)


def prewitt_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)

    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)

    edges = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))

    return cv2.convertScaleAbs(edges)


def roberts_edge(img):
    gray = to_gray(img)

    kernelx = np.array([[1, 0],
                        [0, -1]], dtype=np.float32)

    kernely = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernely)

    edges = cv2.magnitude(gx, gy)

    return cv2.convertScaleAbs(edges)


def laplacian_edge(img):
    gray = to_gray(img)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    lap = cv2.Laplacian(blurred, cv2.CV_64F)

    return cv2.convertScaleAbs(lap)


def log_edge(img, ksize=5, sigma=1.0):
    gray = to_gray(img)

    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    log = cv2.Laplacian(blurred, cv2.CV_64F)

    return cv2.convertScaleAbs(log)