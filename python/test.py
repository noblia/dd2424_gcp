from plotting import plot_confusion_matrix, plot_f1_scores
from dataset import get_data
from itertools import groupby, product
from Network import Network

from os.path import join,dirname
from random import choice, shuffle
from sklearn.metrics import confusion_matrix, f1_score
from skimage import color
from sys import argv, exit

'''Main file which gets data from a directory, runs it through the
network, gets the predictions from the network based on test data then
plots confusion matrix and bar chart of f1_scores
'''

if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)

x_train, x_test, y_train, y_test = get_data(argv[1], 5000)

net = Network(n_epochs = 120)
net.train_network(x_train, y_train)

val = net.pred_network(x_test, y_test)

#get predictions from network
y_guesses = [el['classes'] for el in val]

plot_f1_scores(y_test, y_guesses)

mat = confusion_matrix(y_test, y_guesses).T
plot_confusion_matrix(mat)
