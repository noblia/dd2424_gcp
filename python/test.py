from dataset import cells_in_dataset
from Network import Network
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from skimage import color
from sys import argv, exit

if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)

cells = list(zip(*cells_in_dataset(argv[1])))
xt = np.float32(cells[0])
labels = np.array(cells[1])

net = Network(n_epochs=1000)
net.train_network(xt, labels)
print(net.eval_network(xt, labels))
