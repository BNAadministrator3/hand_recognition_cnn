import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import optimizers
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

model = load_model('h5files/conv6_gap_dataaug_150epoches.h5')
optimizer = optimizers.Adadelta()
model.compile(optimizer=optimizer, loss=categorical_crossentropy,metrics=['accuracy'])

from cnn_utils import *
import gc
_, _, X_test_orig, Y_test_orig, classes = load_dataset()
#normlize the data into the [0,1]
X_test = X_test_orig / 255.
del X_test_orig
gc.collect()

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred,axis=1)
a=1
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # temp = unique_labels(y_true, y_pred)
    # classes = classes[temp]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

class_names = ['0','1','2','3','4','5']
y_test = Y_test_orig[0]
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()