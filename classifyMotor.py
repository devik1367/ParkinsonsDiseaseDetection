import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import collections
from math import sqrt
import matplotlib.pyplot as plt


# main function: find threshold value
def find_threshold(X_train, X_test, y_train, y_test):
    # perform feature selection
    X_train, X_test = preprocessing.random_forest_features(X_train, y_train, X_test)

    # create dictionary to hold mcc values for different thresholds
    mcc_values = collections.defaultdict()
    accuracy_values = []

    for threshold in range(10, 35):
        # split motor updrs based on threshold
        y_train1, y_test1 = np.array(y_train), np.array(y_test)
        y_train1, y_test1 = list(np.where(y_train1 > threshold, 1, 0)), list(np.where(y_test1 > threshold, 1, 0))

        # classify input patterns using data
        y_pred = list(mlp(X_train, y_train1, X_test))

        # calculate prediction accuracy
        accuracy_values.append(accuracy_score(y_test1, y_pred))

        # calculate MCC
        mcc_values[threshold] = mcc(y_test1, y_pred)

        # find threshold value that gives highest mcc value
        maxMCC = 0
        # iterate through (k = threshold, v = mcc) pairs
        for key, value in mcc_values.items():
            if value > maxMCC:
                maxMCC = value
                thresh = key

    # plot threshold values against MCC
    plt.plot(list(mcc_values.keys()), list(mcc_values.values()))
    plt.xticks(list(mcc_values.keys()))
    plt.xlabel("Motor-UPDRS Threshold")
    plt.ylabel("Matthews Correlation Coefficient")
    plt.savefig('thresholdVSmcc.png')

    # plot threshold values against model accuracy
    plt.clf()
    plt.plot(list(mcc_values.keys()), accuracy_values, color="green")
    plt.xticks(list(mcc_values.keys()))
    plt.xlabel("Motor-UPDRS Threshold")
    plt.ylabel("prediction Accuracy")
    plt.savefig('thresholdVSaccuracy.png')
    return thresh


# MLP binary classifier
def mlp(X_train, y_train, X_test):
    # build model
    model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                          random_state=1)

    # train data
    model.fit(X_train, y_train)

    # test data and generate predictions
    y_pred = model.predict(X_test)
    return y_pred


# calculate matthews correlation coefficient
def mcc(yt, yp):
    labels = [0, 1]
    cm = confusion_matrix(yt, yp, labels).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if x != 0:
        result = ((tp * tn) - (fp * fn)) / sqrt(x)
    else:
        # if any of the four sums is 0, arbitrarily set denominator to 1
        # see https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        result = ((tp * tn) - (fp * fn))
    return result
