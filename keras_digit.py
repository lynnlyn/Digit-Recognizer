from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np


def data_label(data):
    x = []
    y = []
    for row in data.itertuples(index=True, name='Pandas'):
        image = np.array(row[2::]).reshape(28, 28, 1)
        x.append(image)
        y.append(row[1])
    y = label_reshape(y)
    return np.array(x), y


def label_reshape(label):
    labels_ = []
    I = np.eye(10, dtype=np.float32)
    for i in label:
        labels = I[i,:]
        labels_.append(labels)
    return np.array(labels_)


if __name__ == '__main__':

    train_csv = r'/Users/Lynn/Desktop/Digit Recognizer/datasets/train.csv'
    data = pd.read_csv(train_csv)
    m = len(data)
    train_data = data.iloc[0:int(len(data)/2), :]
    test_data = data.iloc[int(len(data)/2)::, :]

    X_train, Y_train = data_label(train_data)
    X_test, Y_test = data_label(test_data)

    nb_classes = 10
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (28,28,1)

    model = Sequential()
    model.add(Conv2D(nb_filters,(kernel_size[0], kernel_size[1]),padding='same',input_shape=input_shape))
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

    model.fit(X_train, Y_train, batch_size=500, epochs=50,
              verbose=1, validation_data=(X_test, Y_test))


