import numpy as np
import cv2


def add_noise(img, noise_type="gaussian"):
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(0, 25, img.shape)
        img = img + noise

    elif noise_type == "salt_pepper":
        prob = 0.02
        salt = np.random.rand(*img.shape[:2]) < prob
        pepper = np.random.rand(*img.shape[:2]) < prob

        if len(img.shape) == 3:
            img[salt] = 255
            img[pepper] = 0
        else:
            img[salt] = 255
            img[pepper] = 0

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_filter(img, filter_type="blur"):
    if filter_type == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)

    elif filter_type == "median":
        return cv2.medianBlur(img, 5)

    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    return img
