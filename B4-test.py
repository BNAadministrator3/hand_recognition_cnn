from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
from cnn_utils import *
import gc
import os
import time
import matplotlib.pyplot as plt

CLASS_NUM = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only display error and warning; for 1: all info; for 3: only error.

class CNNmodels():
    def __init__(self):
        pass
    def baselineCNN(self,input_shape):
        X_input = Input(name='the_input', shape=input_shape)
        h = Conv2D(kernel_size=3, filters=32, strides=1, kernel_initializer='he_normal',activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X_input)
        h = Conv2D(kernel_size=3, filters=32, strides=1, kernel_initializer='he_normal',activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(h)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
        h = Conv2D(kernel_size=3, filters=64, strides=1, kernel_initializer='he_normal',activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(h)
        h = Conv2D(kernel_size=3, filters=64, strides=1, kernel_initializer='he_normal',activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(h)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
        h = Conv2D(kernel_size=3, filters=128, strides=1, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.0005))(h)
        h = Conv2D(kernel_size=3, filters=128, strides=1, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.0005))(h)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
        flayer = GlobalAveragePooling2D()(h)
        fc2 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)
        model = Model(inputs=X_input, outputs=y_pred)
        return model
    # def baselineCNN(self,input_shape):
    #     X_input = Input(name='the_input', shape=input_shape)
    #     h = Conv2D(kernel_size=3, filters=32, strides=1, kernel_initializer='he_normal',activation='relu', padding='same')(X_input)
    #     h = Conv2D(kernel_size=3, filters=32, strides=1, kernel_initializer='he_normal',activation='relu', padding='same')(h)
    #     h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
    #     # h = Conv2D(kernel_size=3, filters=64, strides=1, kernel_initializer='he_normal',activation='relu', padding='same')(h)
    #     # h = Conv2D(kernel_size=3, filters=64, strides=1, kernel_initializer='he_normal',activation='relu', padding='same')(h)
    #     # h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
    #     # h = Conv2D(kernel_size=3, filters=128, strides=1, kernel_initializer='he_normal', padding='same')(h)
    #     # h = Conv2D(kernel_size=3, filters=128, strides=1, kernel_initializer='he_normal', padding='same')(h)
    #     # h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)  # 池化层
    #     # flayer = GlobalAveragePooling2D()(h)
    #     flayer = Flatten()(h)
    #     fc2 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
    #     y_pred = Activation('softmax', name='Activation0')(fc2)
    #     model = Model(inputs=X_input, outputs=y_pred)
    #     return model
    def modelcompiler(self,model):
        optimizer = optimizers.Adadelta()
        # optimizer = optimizers.SGD(lr=0.01)
        model.compile(optimizer=optimizer, loss=categorical_crossentropy,metrics=['accuracy'])
        return model

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = loadup_dataset()
#normlize the data into the [0,1]
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
del X_train_orig, X_test_orig
gc.collect()
Y_train = convert_to_one_hot(Y_train_orig, CLASS_NUM).T
Y_test = convert_to_one_hot(Y_test_orig, CLASS_NUM).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

#form data generator
from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    # horizontal_flip=True,
)
batch_size = 64
# train_generator = gen.flow(X_train,
#                            Y_train,
#                         batch_size=batch_size)
train_generator = gen.flow_from_directory(
        'datasets/bmp/train',
        target_size=(64, 64),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True) #Why load from file works badly??


#form CNN
s = CNNmodels()
inshape = [64,64,3]
model = s.baselineCNN(inshape)
model = s.modelcompiler(model)
model.summary()
epochs = 100
history = AccuracyHistory()

#train it
model.load_weights('./h5files/newdata_conv6_gap_dataaug_200epoches_weigths.h5')
st = time.time()
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, shuffle=True, callbacks=[history])
model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//batch_size+1, epochs=epochs,  callbacks=[history])
print('Training time: {}s'.format(round(time.time()-st,2)))

# score = model.evaluate(X_train, Y_train, verbose=0)
# print('Train loss:', score[0])
# print('Train accuracy:', score[1])
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
train_generator = gen.flow_from_directory(
        'datasets/bmp/train',
        target_size=(64, 64),
        batch_size=1,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True) #Why load from file works badly??

test_generator = gen.flow_from_directory(
        'datasets/bmp/test',
        target_size=(64, 64),
        batch_size=1,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=False) #Why load from file works badly??

score = model.evaluate_generator(generator=train_generator)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate_generator(generator=test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# model.save('model/newdata_conv6_gap_dataaug_200epoches.h5')
# model.save_weights('model/newdata_conv6_gap_dataaug_200epoches_weigths.h5')

plt.plot(range(epochs), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()