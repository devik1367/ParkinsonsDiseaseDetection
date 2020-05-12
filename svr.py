from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


def svr_model(X_train, y_train, X_test, y_test):

    # create model using different kernels
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

    # fit model on training data
    svr_rbf.fit(X_train, y_train)

    # make predictions on test data
    y_pred = svr_rbf.predict(X_test)

    # print R squared score
    score = svr_rbf.score(X_train, y_train)

    # calculate root mean squared error
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    mae = math.sqrt(mean_absolute_error(y_test, y_pred))

    return score, rmse, mae
