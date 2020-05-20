
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from random import randint
from sklearn import metrics
from skimage.feature import hog

train_csv = r'/Users/Lynn/Desktop/Digit Recognizer/datasets/train.csv'
test_csv = r'/Users/Lynn/Desktop/Digit Recognizer/datasets/test.csv'

data = pd.read_csv(train_csv)
train_data = data.iloc[0:int(len(data)/2), :]
data_size = len(train_data)
train_labels = np.zeros(shape=(data_size, 1))
model = RandomForestClassifier(oob_score=True, random_state=10)

batch_size = 500
m = int(data_size/batch_size) - 1

for i in range(m):
    train_patch = train_data.iloc[batch_size*i:batch_size*(i+1), :]

    image_set = []
    label_set = []
    feature_set = []

    for row in train_patch.itertuples(index=True, name='Pandas'):

        t_label_ = row[1]
        label_set.append(t_label_)

        t_image_ = np.array(row[2::])
        t_image_ = t_image_.reshape(28, 28)
        #t_image_ = cv2.cvtColor(t_image_, cv2.COLOR_BGR2GRAY)
        image_set.append(t_image_)
        fd, hog_image = hog(t_image_, orientations=8, pixels_per_cell=(2, 2),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
        counts, bins = np.histogram(fd)
        feature_set.append(counts)
        # plt.figure(1)
        # plt.title(t_label_)
        # plt.imshow(t_image_)
        # plt.figure(2)
        # plt.title(t_label_)
        # plt.imshow(hog_image)
        # plt.figure(3)
        # plt.hist(fd, 8)
        # plt.show()

    label_set = np.array(label_set)
    feature_set = np.array(feature_set)
#     image_set = np.array(image_set)
#
    model.fit(feature_set, label_set)
    print("accuracy:%f" % model.oob_score_)

#
#     #print('test')
#
# test_batch_size = 500
# test_data = data.iloc[int(len(data)/2)::, :]
# test_data_size = len(test_data)
# n = int(test_data_size/test_batch_size) - 1
#
# for i in range(n):
#     test_patch = test_data.iloc[test_batch_size*i:test_batch_size*(i+1), :]
#     test_images = []
#     test_labels = []
#
#     for row in test_patch.itertuples(index=True, name='Pandas'):
#         test_image = np.array(row[2::])
#         test_images.append(test_image)
#
#         label = row[1]
#         test_labels.append(label)
#
#     test_images = np.array(test_images)
#     test_labels = np.array(test_labels)
#     pre_label = model.predict(test_images)
#     acc_test = metrics.accuracy_score(test_labels, pre_label)
#
#     print('Test data accuracy: ', acc_test)


