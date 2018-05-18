from dataset import cells_in_dataset
from Network import Network
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from random import shuffle
from skimage import color
from sys import argv, exit

if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)

xy_pairs = list(cells_in_dataset(argv[1], [0.35, 0.25, 0.31, 0.09]))
shuffle(xy_pairs)

# Transpose
cells = list(zip(*xy_pairs))

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

