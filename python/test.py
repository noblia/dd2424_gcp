from dataset import cells_in_dataset
# from Network import Network
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from random import shuffle
from skimage import color
from sys import argv, exit

if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)
# [0.35, 0.25, 0.31, 0.09]
xy_pairs = list(cells_in_dataset(argv[1], [0.35/0.35, 0.25/0.35, 0.31/0.35, 0.09/0.35]))

shuffle(xy_pairs)

# Transpose
cells = list(zip(*xy_pairs))
# print(cells[1].count(0), cells[1].count(1), cells[1].count(2), cells[1].count(3))


x = np.float32(cells[0])
y = np.array(cells[1])

# 80% train, 20% test
split = int(len(xy_pairs)*0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

net = Network(n_epochs=120)
net.train_network(x_train, y_train)

val = net.eval_network(x_test, y_test)
r = val['recall']
p = val['precision']
f1 = (2 * p * r) / (p + r)
print(val, f1)

