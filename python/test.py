from dataset import cells_in_dataset
from itertools import groupby
from Network import Network
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from random import choice, shuffle
from sklearn.metrics import confusion_matrix, f1_score
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if len(argv) != 2:
    print('usage %s: datadir' % argv[0])
    exit(1)
cls_samp = 5000
yx = list(cells_in_dataset(argv[1]))
yx = list(uniform_class_sampling(yx, cls_samp))
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


val = net.pred_network(x_test,y_test)
y_guesses = [el["classes"] for el in val]

print(confusion_matrix(y_test, y_guesses).T)

f1_classes = f1_score(y_test,y_guesses, average=None)
f1_averages=  f1_score(y_test,y_guesses, average = 'weighted')

print(y_guesses)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
