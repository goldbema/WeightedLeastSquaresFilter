import cv2
import numpy as np


def rgb_to_lab(img):
    if img.ndim != 3:
        raise ValueError("Image must contain 3 channels")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return lab


def sigmoid(img, a):
    return 1. / (1. + np.exp(-a * img.astype(np.float64)))


# Normalized sigmoid function inspired by the authors' implementation, available here:
# http://www.cs.huji.ac.il/~danix/epd/msdm-example.zip

def normalized_sigmoid(img, a):
    # Note that image must be normalized to the range [-0.5, 0.5]
    centered_sigmoid = sigmoid(img, a) - 0.5

    # Scale domain and range endpoints to be identical 
    # NOTE: Don't do this. It causes low luminance values to take
    # on values near the luminance mean.
    scaled_sigmoid = centered_sigmoid #* (0.5 / sigmoid(np.ones_like(img, dtype=np.float64) * 0.5, a))

    return scaled_sigmoid


# This algorithm for edge-finding is due to Winnemoller et. al (2006) ("Real-Time Video Abstraction").
def difference_of_gaussians(img, sigma_e=5, tau=0.98, steepness=0.75):
    if img.ndim != 2:
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)

    sigma_r = sigma_e * np.sqrt(1.6)

    S_e = cv2.GaussianBlur(img, (0,0), sigma_e)
    S_r = cv2.GaussianBlur(img, (0,0), sigma_r)

    difference = S_e - (tau * S_r)
    falloff = 1 + np.tanh(difference * steepness)

    D = np.ones_like(img, dtype=np.float64)

    D[difference <= 0] = falloff[difference <= 0]

    return D
