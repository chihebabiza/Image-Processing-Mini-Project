import numpy as np
import cv2

def apply_filter(img, filter_type="mean", ksize=5, sigma=1.0):

    # enforce odd kernel size (VERY IMPORTANT for OpenCV)
    if ksize % 2 == 0:
        ksize += 1

    if filter_type == "mean":
        return cv2.blur(img, (ksize, ksize))

    elif filter_type == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    elif filter_type == "median":
        return cv2.medianBlur(img, ksize)

    else:
        raise ValueError("filter_type must be 'mean', 'gaussian', or 'median'")