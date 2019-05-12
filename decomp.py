import cv2
import numpy as np

import utils
import wls


def generate_decomposition(alpha_schedule=[1.2, 1.2, 1.2], lambda_schedule=[0.1, 0.8, 6.4], method='progressive'):
    if len(alpha_schedule) == 0 or len(lambda_schedule) == 0:
        raise ValueError('lambda_schedule length must be positive')
    if len(alpha_schedule) != len(lambda_schedule):
        raise ValueError('schedules must be of identical length')

    def decompose(L):
        L_prev = L.astype(np.float64)
        sequence = []

        # Process difference images in sequence
        for alpha, lambda_ in zip(alpha_schedule, lambda_schedule):
            L_cur = L if method == 'progressive' else L_prev

            L_next = wls.wls_filter(L_cur, lambda_=lambda_, alpha=alpha)

            # Obtain and append difference image
            D = L_prev - L_next
            sequence.append(D)

            L_prev = L_next

        # Append coarsest image to the sequence as the base image
        sequence.append(L_next)

        return sequence

    return decompose


def reconstitute(sequence):
    reconstituted = []

    img_sum = np.zeros_like(sequence[0], dtype=np.float64)
    for s in reversed(sequence):
        img_sum += s
        reconstituted.append(img_sum.copy())

    reconstituted = list(reversed(reconstituted))
    return reconstituted
