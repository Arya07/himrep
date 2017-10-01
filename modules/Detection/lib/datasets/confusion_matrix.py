print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(class_names,results_file_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    class_names=class_names+('nothing',)
    classes=[]
    for j in range(0, len(class_names)-1):
        classes.append(class_names[j])
    # Extract y_test and y_pred
    y_test = []
    y_pred = []

    with open(results_file_name, 'r') as f:
        # example of a line in f:
        # book/book5/MIX/day5/left/00005559 book5
        for line in f.readlines():
            line = line.strip('\n')
            split_test = line.split('/')
            split_pred = line.split(' ')
            y_test.append(split_test[1])
            if split_pred[1]=='' or split_pred[1]==' ':
                y_pred.append('nothing')
            else:
                y_pred.append(split_pred[1])

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    with open('labels.txt', 'w+') as f:
        f.write('test {:.4f}  pred {:.4f}\n'.format(len(y_test), len(y_pred)))
        for i in range(0, len(y_test)-1):
            f.write('{} {}\n'.format(y_test[i], y_pred[i]))

    # Compute confusion matrix
    # print y_test
    # print y_pred
    print classes
    cm = confusion_matrix(y_test, y_pred, classes)
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm=np.around(cm, decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Plot non-normalized confusion matrix
    plt.figure()
    # plot_confusion_matrix(cm, classes=class_names,
    #                       title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cm, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()
