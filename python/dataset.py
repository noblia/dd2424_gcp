import numpy as np
import scipy.io as sio
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from itertools import groupby, product
import matplotlib.pyplot as plt
from math import isnan
from os.path import join
from random import randint, random, choice, shuffle
from sys import argv, exit
from Image import Image
from SubImage import SubImage

'''
Input: cell image
Output: cell image randomly perturbed in HSV space then converted back to RGB space
'''
def perturbe_color(sel):
    sel = rgb2hsv(sel)
    r = [
        np.random.uniform(low = 0.95, high = 1.05),
        np.random.uniform(low = 0.9, high = 1.1),
        np.random.uniform(low = 0.9, high = 1.1)]
    for i in range(3):
        sel[:, :, i] = np.clip(sel[:, :, i] * r[i], 0, 1)
    return hsv2rgb(sel)
'''
Input: cell imgae
Output: cell image randomly flipped around either horizontal or vertical axis. Can also be returned unperturped
'''
def flip(sel):
    r = randint(0, 3)
    sel2 = sel
    if r == 0:
        sel2 = np.fliplr(sel)
    elif r == 1:
        sel2 = np.flipud(sel)
    elif r == 2:
        sel2 = sel2
    return sel2

'''
Input: cell image
Output: rotated cell image, either 0, 90, 180 or 270 degrees
'''
def rotate(sel):
    r = randint(0,4)
    return np.rot90(sel, k=r)

'''
Input: a raw image and the coordinates of a cell in the picture
Output: an image of size 27x27 pixels and with the cell at the center
'''
def subImage(img, px, py):
    s_x = int(round(px - 27.0 / 2))
    e_x = s_x + 27
    s_y = int(round(py - 27.0 / 2))
    e_y = s_y + 27
    return img[s_y:e_y, s_x:e_x, :]

'''
Input: list of tuples, where the tuples are cell image data with  corresponding label
Output: n cells and corresponding labels uniformly sampled from input list
'''
def uniform_class_sampling(yx, n):
    gb = groupby(sorted(yx, key = lambda el: el[0]),
                 key = lambda el: el[0])
    gb = [(k, list(v)) for (k, v) in gb]
    for k, lst in gb:
        for x in range(n):
            yield (k, choice(lst)[1])

'''
Input: directory for raw data, and idx corresponding to one raw data image file
Output: All raw cell images from one image folder and their corresponding labels
'''
def cells_in_image(base_dir, idx):
    base_path = join(base_dir, 'img%d/img%d' % (idx, idx))
    image_path = base_path + '.bmp'

    img = plt.imread(image_path)
    classes = ['epithelial', 'fibroblast', 'inflammatory', 'others']

    # iterate through all classes in one image folder
    for cls_idx, cls in enumerate(classes):
        cls_path = base_path + '_' + cls + '.mat'
        mat = sio.loadmat(cls_path)['detection'].reshape(-1, 2)
        for [px, py] in mat:
            sel = subImage(img, px, py)
            if sel.shape != (27, 27, 3):
                continue
            yield cls_idx, sel

'''
Input: directory for raw data
Output: tuple of cell images and their corresponding labels
'''
def cells_in_dataset(base_dir):
    for x in range(1, 101):
        for tup in cells_in_image(argv[1], x):
            yield tup

def perform_perturbation(yx):
    for cls_idx, img in yx:
        img = perturbe_color(rotate(flip(img)))
        img = rgb2gray(img)
        yield cls_idx, img.reshape(-1)


'''
Input: directory for raw data, number of wanted samples per class
Output: training and test data
'''
def get_data(base_dir, cls_samp):
    yx = list(cells_in_dataset(base_dir))

    yx = list(uniform_class_sampling(yx, cls_samp))

    yx = list(perform_perturbation(yx))
    shuffle(yx)

    cells = list(zip(*yx))

    x = np.float32(cells[1])
    y = np.array(cells[0])

    # 80% train, 20% test
    split = int(len(yx)*0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    return x_train, x_test, y_train, y_test
