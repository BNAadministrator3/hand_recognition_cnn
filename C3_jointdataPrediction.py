import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import optimizers
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

model = load_model('h5files/jointdata_conv6_gap_dataaug_200epoches.h5')
optimizer = optimizers.Adadelta()
model.compile(optimizer=optimizer, loss=categorical_crossentropy,metrics=['accuracy'])

from cnn_utils import *
import gc
# Loading the data (signs)
X_train_orig_blue, Y_train_orig_blue, X_test_orig_blue, Y_test_orig_blue, classes = loadup_dataset()
#normlize the data into the [0,1]
X_train_blue = X_train_orig_blue / 255.
X_test_blue = X_test_orig_blue / 255.
del X_train_orig_blue, X_test_orig_blue
gc.collect()

X_train_orig_red, Y_train_orig_red, X_test_orig_red, Y_test_orig_red, classes = load_dataset()
#normlize the data into the [0,1]
X_train_red = X_train_orig_red / 255.
X_test_red = X_test_orig_red / 255.
del X_train_orig_red, X_test_orig_red
gc.collect()

X_train = np.concatenate((X_train_blue,X_train_red))
X_test = np.concatenate((X_test_blue,X_test_red))
Y_train_orig = np.concatenate((Y_train_orig_blue,Y_train_orig_red),axis=1)
Y_test_orig = np.concatenate((Y_test_orig_blue,Y_test_orig_red),axis=1)
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


#retest
score = model.evaluate(X_train, Y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('')
print('Then use model.prediction')
Y_pred = model.predict(X_train)
y_pred = np.argmax(Y_pred,axis=1)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_train_orig[0], y_pred))


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

class_names = ['0','1','2','3','4','5','6','7','8','9']
y_test = Y_test_orig[0]
plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')

plt.show()