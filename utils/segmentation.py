import cv2
import numpy as np


def segmentation_kmeans(img, k=3):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    Z = img_lab.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img_lab.shape)

    return cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
