import matplotlib

matplotlib.use('TkAgg')

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm, metrics
import numpy as np
import tensorflow.contrib.learn.python.learn as learn


digits = load_digits()

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target)

n_classes = len(set(Y_train))
classifier = learn.LinearClassifier(n_classes=n_classes, feature_columns=learn.infer_real_valued_columns_from_input(digits.target))
classifier.fit(X_train, Y_train, steps=40)

y_pred = list(classifier.predict(X_test))

print(metrics.classification_report(y_true=Y_test, y_pred=y_pred))

# classifier = svm.SVC(gamma=0.001)
# classifier.fit(X_train, Y_train)
# y_pred = classifier.predict(X_test)
#
# print(metrics.classification_report(y_true=Y_test, y_pred=y_pred))


# print(1.0 - abs(np.mean(Y_test - predicted)))

# fig = plt.figure(figsize=(3,3))
#
# plt.imshow(digits['images'][66], cmap='gray', interpolation='none')
#
# plt.show()