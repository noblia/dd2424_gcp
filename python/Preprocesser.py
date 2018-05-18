import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv


class Preprocesser:

    def __init__(self, dataset):
        # Dataset is in hsv space
        self.dataset = dataset

    def rotate(self, degree, img):
        return np.rot90(img, k=round(degree/90))


    def flip(self, axis, img):
        if axis == 0:
            img = np.fliplr(img)
        if axis == 1:
            img = np.flipup(img)
        return img


    def perturbe_color(self, h_limits=(0.95, 1.05), sv_limits=(0.9, 1.1)):
        # r = (r_h, r_s, r_v)
        dataset = np.zeros(self.dataset.shape)
        r = (
            np.random.uniform(low=h_limits[0], high=h_limits[1]),
            np.random.uniform(low=sv_limits[0], high=sv_limits[1]),
            np.random.uniform(low=sv_limits[0], high=sv_limits[1]))
        i = 0
        for n in r:
            self.dataset[:, :, i, :] = self.dataset[:, :, i, :] * n
            i += 1

