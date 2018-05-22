from dataset import cells_in_dataset
from itertools import groupby
from Network import Network
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from random import choice, shuffle
from skimage import color
from sys import argv, exit

def uniform_class_sampling(yx, n):
    gb = groupby(sorted(yx, key = lambda el: el[0]),
                 key = lambda el: el[0])
    gb = [(k, list(v)) for (k, v) in gb]
    print(len(gb[0][1]))
    for k, lst in gb:
        for x in range(n):
            yield (k, choice(lst)[1])


if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)

yx = list(cells_in_dataset(argv[1]))
yx = list(uniform_class_sampling(yx, 5000))
shuffle(yx)

# Transpose
cells = list(zip(*yx))

x = np.float32(cells[1])
y = np.array(cells[0])

# 80% train, 20% test
split = int(len(yx)*0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

net = Network(n_epochs=120)
net.train_network(x_train, y_train)

val = net.eval_network(x_test, y_test)
r = val['recall']
p = val['precision']
f1 = (2 * p * r) / (p + r)
if [p,r] == 0:
    f1 = 'undefined'

print(val, f1)

