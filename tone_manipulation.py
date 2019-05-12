'''
An implementation of:

    Farbman, Z., Fattal, R., Lischinski, D. and Szeliski, R., 
    "Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation,"
    ACM Transactions on Graphics, vol. 27, no. 3, Aug. 2008.

'''

__author__ = 'Maxwell Goldberg'

import argparse
import cv2
import numpy as np

import decomp
import utils


def tone_manipulation(lab, sequence, a_values, L_min=0, L_max=255, exposure=1.0, saturation=1.0, mask=None):
    lab = lab.astype(np.float64)

    L = lab[..., 0]
        
    L_mean = np.mean(L)

    # Normalize the sequence to [-0.5, 0.5] by compressing it between the min and max luminance values
    # and subtracting the luminance mean from the base layer.
    normalized_sequence = []
    for s, L in enumerate(sequence):
        L = L.copy()
        if s == (len(sequence) - 1):
            L -= L_mean
        L_normalized = (L - L_min) / (L_max - L_min)
        normalized_sequence.append(L_normalized)

    # Apply the normalized sigmoid nonlinearity. The sequence range is still [-0.5, 0.5]
    nonlinear_sequence = []
    for i, (L_normalized, a) in enumerate(zip(normalized_sequence, a_values)):
        if i == len(normalized_sequence)-1:
            L_normalized *= exposure
        nonlinearity = utils.normalized_sigmoid(L_normalized, a)
        nonlinear_sequence.append(nonlinearity)


    # Rescale the normalized sequence back to its original range.
    denormalized_sequence = []
    for n, nonlinear_L in enumerate(nonlinear_sequence):
        nonlinear_L = nonlinear_L * (L_max - L_min) + L_min
        if n == (len(sequence) - 1):
            nonlinear_L += L_mean
        denormalized_sequence.append(nonlinear_L)

    # Form the new luminance image. Apply the detail mask to each layer (if it exists).
    L_new = np.zeros_like(L, dtype=np.float64)
    for d, denormalized_L in enumerate(denormalized_sequence):
        if mask is not None:
            denormalized_L[mask < (len(sequence) - d - 1)] = 0
        L_new += denormalized_L
    L_new = np.clip(L_new, a_min=L_min, a_max=L_max)

    lab[..., 0] = L_new

    lab[..., 1:] = (lab[..., 1:] - 128) * np.float64(saturation) + 128

    return lab


# This application is inspired by the trackbar application in the OpenCV documentation 
# available here: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html

class ToneManipulationApplication(object):
    def __init__(self, img_path, alpha_schedule, lambda_schedule, method='progressive'):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError('Invalid image path: {}'.format(img_path))
        
        self.dirty = False
        self.show_lines = False

        self.a_values = np.ones(len(alpha_schedule) + 1, dtype=np.float64)
        self.exposure = 1.0
        self.saturation = 1.0

        self.decompose = decomp.generate_decomposition(alpha_schedule=alpha_schedule,
                                                  lambda_schedule=lambda_schedule,
                                                  method=method)
        self.sequence = None
        self.mask = None
        lab = self._manipulate_tone()
        lab = np.round(lab).astype(np.uint8)
        self.vis = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        self.color = len(alpha_schedule)
        self.mask = np.ones_like(self.vis[:,:,0]) * self.color
        self.drawing = False
        self.radius = 100
        self.x, self.y = 0, 0

        self._init_ui()


    def gen_trackbar_fn(self, idx=None, exposure=None, saturation=None):
        def trackbar_fn(x):
            self.dirty = True
            if idx is not None:
                self.a_values[idx] = float(x)
            elif exposure is not None:
                self.exposure = x / 100.
            elif saturation is not None:
                self.saturation = x / 100.
        return trackbar_fn


    def on_mouse_event(self, event, x, y, flags, params):
        self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.dirty = True

        if self.drawing:
            cv2.circle(self.mask, (x,y), self.radius, self.color, -1)
            

    def _init_ui(self):
        self.window_name = 'image'
        cv2.namedWindow(self.window_name)

        trackbar_min, trackbar_max = 0, 50
        cv2.createTrackbar('base', self.window_name, trackbar_min, trackbar_max, self.gen_trackbar_fn(idx=-1))
        cv2.setTrackbarPos('base', self.window_name, 1)
        for i in range(1,len(self.sequence)):
            cv2.createTrackbar('detail_{}'.format(i), 
                               self.window_name, trackbar_min, trackbar_max, 
                               self.gen_trackbar_fn(idx=len(self.sequence)-i-1))
            cv2.setTrackbarPos('detail_{}'.format(i), self.window_name, 1)

        cv2.createTrackbar('exposure', self.window_name, 0, 200, self.gen_trackbar_fn(exposure=True))
        cv2.setTrackbarPos('exposure', self.window_name, 100)

        cv2.createTrackbar('saturation', self.window_name, 0, 200, self.gen_trackbar_fn(saturation=True))
        cv2.setTrackbarPos('saturation', self.window_name, 100)

        cv2.setMouseCallback('image', self.on_mouse_event)


    def _manipulate_tone(self):
        lab = utils.rgb_to_lab(self.img)
        L = lab[..., 0]
            
        L_mean = np.mean(L)

        if self.sequence is None:
            self.sequence = self.decompose(L)

        lab = tone_manipulation(lab, 
                                self.sequence, 
                                self.a_values, 
                                exposure=self.exposure, 
                                saturation=self.saturation, 
                                mask=self.mask)
        return lab


    def run(self):
        while True:
            vis = self.vis.copy()
            cv2.circle(vis, (self.x,self.y), self.radius, (0,0,255), 1)
            cv2.imshow(self.window_name, vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                self.show_lines = not self.show_lines
                self.dirty = True
            elif k in [ord('0'), ord('1'), ord('2'), ord('3')]:
                self.color = k - 48
            elif k == ord('d'):
                self.radius = min(self.radius + 2, 100)
            elif k == ord('a'):
                self.radius = max(self.radius - 2, 2)
            elif k == ord('s'):
                cv2.imwrite('output.jpg', self.vis)
            elif k == 27:
                break

            if self.dirty:
                self.dirty = False
                lab = self._manipulate_tone()
                if self.show_lines:
                    D = utils.difference_of_gaussians(lab[..., 0], sigma_e=1.5, steepness=2)
                    lab[..., 0] *= D.astype(np.float64)
                lab = np.round(lab).astype(np.uint8)
                self.vis = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Change tone and detail properties of an image')
    parser.add_argument('--img_path', type=str, required=True, help='Path to image')
    args = parser.parse_args()

    return args


def main(args):
    ALPHA_SCHEDULE = [2.0, 2.0, 2.0] 
    LAMBDA_SCHEDULE = [0.05, 0.2, 0.8]   
    METHOD = 'progressive'

    ToneManipulationApplication(args.img_path, 
                                ALPHA_SCHEDULE, 
                                LAMBDA_SCHEDULE, 
                                METHOD).run()


if __name__ == '__main__':
    args = parse_args()
    main(args)