from itertools import groupby, product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

labels = ['epithelial', 'fibroblast', 'inflammatory', 'misc.']

'''Function which plots a confusion matrix. Source:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

Input: confusion matrix, class labels, boolean to choose normalization
or not, color scheme of plot Output: figure of plotted confusion
'''
def plot_confusion_matrix(cm, normalize = True, cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.savefig('/home/matilda.noblia/dd2424_gcp/python/figures/mat.png')

def plot_f1_scores(y_true, y_guesses):
    scores = list(f1_score(y_true, y_guesses, average = None))
    avg = f1_score(y_true, y_guesses, average = 'weighted')
    scores.append(avg)

    labels2 = labels + ['average']
    n_bars = len(scores)

    plt.figure()
    plt.bar(np.arange(n_bars), scores)
    tick_marks = np.arange(n_bars)
    plt.xticks(tick_marks, labels2)
    plt.savefig('/home/matilda.noblia/dd2424_gcp/python/figures/bar.png')
