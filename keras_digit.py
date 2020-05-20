#-*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
import os


class DigitRecogniter:

    def __init__(self, data, model_para):

        self.data = data
        self.model_para = model_para

        self.images = None
        self.labels = None

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        self.model = None

    def preprocess_data(self):
        images = []
        labels = []
        labels_= []
        for row in self.data.itertuples(index=True, name='Pandas'):
            image = np.array(row[2::]).reshape(28, 28, 1)
            images.append(image)
            labels.append(row[1])

        I = np.eye(10, dtype=np.float32)
        for i in labels:
            label = I[i, :]
            labels_.append(label)

        self.images = np.array(images)
        self.labels = np.array(labels_)

    def load_digits_data(self):

        if len(self.images) != len(self.labels):
            raise ValueError
        else:
            image_num = int(len(self.images)/10*9)
            self.train_data = self.images[0:image_num, :]
            self.test_data = self.images[image_num::, :]

            self.train_label = self.labels[0:image_num, :]
            self.test_label = self.labels[image_num::, :]

    def cnn_model(self):

        nb_classes = self.model_para['classes']
        nb_filters = self.model_para['filters']
        pool_size = self.model_para['pool size']
        kernel_size = self.model_para['kernel size']
        input_shape = self.model_para['shape']

        model = Sequential()
        model.add(
            Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same',
                   input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        model.summary()

        self.model = model

    def train_model(self, batch_size, epochs):

        self.model.fit(self.train_data, self.train_label, batch_size, epochs,
                  verbose=1, validation_data=(self.test_data, self.test_label))


if __name__ == '__main__':

    MODEL_PARA = {
        'classes': 10,
        'filters': 32,
        'pool size': (2, 2),
        'kernel size': (3, 3),
        'shape': (28, 28, 1)}

    data_path = os.getcwd() + '/datasets/train.csv'
    data = pd.read_csv(data_path)
    MNIST = DigitRecogniter(data=data, model_para=MODEL_PARA)
    MNIST.preprocess_data()
    MNIST.load_digits_data()
    MNIST.cnn_model()
    BATCH_SIZE = 200
    EPOCHS = 30
    MNIST.train_model(batch_size=BATCH_SIZE, epochs=EPOCHS)
