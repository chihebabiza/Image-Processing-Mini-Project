import numpy as np
import cv2

def add_noise(img, noise_type="gaussian", mean=0, std=25, prob=0.02):
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(mean, std, img.shape)
        img = img + noise

    elif noise_type == "salt_pepper":
        salt = np.random.rand(*img.shape[:2]) < prob
        pepper = np.random.rand(*img.shape[:2]) < prob

        img[salt] = 255
        img[pepper] = 0

    else:
        raise ValueError("noise_type must be 'gaussian' or 'salt_pepper'")

    return np.clip(img, 0, 255).astype(np.uint8)



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