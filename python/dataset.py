import numpy as np
import scipy.io as sio
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from math import isnan
from os.path import join
from random import randint, random
from sys import argv, exit

from Image import Image
from SubImage import SubImage


class Dataset:
    def __init__(self, imgs_location, no_imgs, sub_img_size):
        self.imgs_location = imgs_location
        self.no_imgs = no_imgs
        self.sub_img_size = sub_img_size
        # list of Image objects
        self.imgs = [None] * self.no_imgs
        self.raw_data = self.getRawData()
        self.dataset = self.dataLoop()

    def dataLoop(self):
        images = []
        for i in range(self.no_imgs):
            # 0 = epithelial

            data_epithelial = self.subImgLoop(self.imgs[i].image, self.imgs[i].epithelial)
            images = np.concatenate((images,
                                     [SubImage(data_epithelial[:, :, :, img], label=0) for img in
                                      range(data_epithelial.shape[-1])]))

            data_fibroblast = self.subImgLoop(self.imgs[i].image, self.imgs[i].fibroblast)
            images = np.concatenate((images, [SubImage(data_fibroblast[:, :, :, img], label=1) for img in
                                              range(data_fibroblast.shape[-1])]))

            data_inflammatory = self.subImgLoop(self.imgs[i].image, self.imgs[i].inflammatory)
            images = np.concatenate((images, [SubImage(data_inflammatory[:, :, :, img], label=2) for img in
                                              range(data_inflammatory.shape[-1])]))

            data_others = self.subImgLoop(self.imgs[i].image, self.imgs[i].others)
            images = np.concatenate(
                (images, [SubImage(data_others[:, :, :, img], label=3) for img in range(data_others.shape[-1])]))

        return images

    def getRawData(self):

        for i in range(self.no_imgs):
            img = Image()
            img_name = 'img' + str(i + 1)
            folder_name = self.imgs_location + img_name + '/' + img_name
            img.image = plt.imread(folder_name + '.bmp').reshape(500, 500, 3)
            img.epithelial = sio.loadmat(folder_name + '_epithelial.mat')['detection'].reshape(-1, 2)
            img.fibroblast = sio.loadmat(folder_name + '_fibroblast.mat')['detection'].reshape(-1, 2)
            img.inflammatory = sio.loadmat(folder_name + '_inflammatory.mat')['detection'].reshape(-1, 2)
            img.others = sio.loadmat(folder_name + '_others.mat')['detection'].reshape(-1, 2)

            self.imgs[i] = img

    def getSubImage(self, img, center):
        border = np.shape(img)[0]

        i = int(round(center[1]))
        j = int(round(center[0]))

        if isnan(i) or isnan(j):
            subImage = np.zeros((self.sub_img_size, self.sub_img_size, 3))

        else:
            offset1, offset2 = self.getOffset(i, border)
            offset3, offset4 = self.getOffset(j, border)

            subImage = img[i - offset1: i + offset2 + 1, j - offset3:j + offset4 + 1, :]

        return subImage

    def subImgLoop(self, img, cell_list):
        no_cells = np.shape(cell_list)[0]
        imgs = np.zeros((self.sub_img_size, self.sub_img_size, 3, no_cells))
        for i in range(no_cells):
            imgs[:, :, :, i] = self.getSubImage(img, cell_list[i, :])

        return imgs

    def getOffset(self, coord, border):
        offset = round((self.sub_img_size - 1) / 2)

        comp = min(coord, border - coord)

        if comp <= offset:
            offset1 = offset - abs(offset - comp) - 1
            offset2 = offset * 2 - offset1
            if coord > border - coord:
                temp = offset1
                offset1 = offset2
                offset2 = temp
        else:
            offset1 = offset
            offset2 = offset

        return offset1, offset2

    def showImage(self, img):
        plt.imshow(img.astype(int))
        plt.show()

def perturbe_color(sel):
    sel = rgb2hsv(sel)
    r = [
        np.random.uniform(low = 0.95, high = 1.05),
        np.random.uniform(low = 0.9, high = 1.1),
        np.random.uniform(low = 0.9, high = 1.1)]
    for i in range(3):
        sel[:, :, i] = np.clip(sel[:, :, i] * r[i], 0, 1)
    return hsv2rgb(sel)

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


def rotate(sel):
    r = randint(0,4)
    return np.rot90(sel, k=r)

def subImage(img, px, py):
    s_x = int(round(px - 27.0 / 2))
    e_x = s_x + 27
    s_y = int(round(py - 27.0 / 2))
    e_y = s_y + 27
    return img[s_y:e_y, s_x:e_x, :]

def cells_in_image(base_dir, idx):
    base_path = join(base_dir, 'img%d/img%d' % (idx, idx))
    image_path = base_path + '.bmp'

    img = plt.imread(image_path)
    classes = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    for cls_idx, cls in enumerate(classes):
        cls_path = base_path + '_' + cls + '.mat'
        mat = sio.loadmat(cls_path)['detection'].reshape(-1, 2)
        for [px, py] in mat:
            sel = subImage(img, px, py)
            if sel.shape != (27, 27, 3):
                continue

            sel = flip(sel)
            sel = rotate(sel)
            # all cells are perturbed in hsv space!!!
            sel = perturbe_color(sel)
            sel = rgb2gray(sel)
            sel = sel.reshape(27, 27)

            yield cls_idx, sel.reshape(-1)

def cells_in_dataset(base_dir):
    for x in range(1, 101):
        for tup in cells_in_image(argv[1], x):
            yield tup


if __name__ == "__main__":
    if len(argv) != 2:
        print('usage %s: datadir' % argv[0])
        exit(1)

    cells = list(zip(*cells_in_dataset(argv[1])))

    obj_arr = np.zeros((2,), dtype=np.object)
    obj_arr[0] = np.array(cells[0]).T
    obj_arr[1] = cells[1]
    sio.savemat('/tmp/out.mat', mdict={'cells': obj_arr})
