import preprocessing
import svr
import classifyMotor


def main():
    # load and preprocess data
    X_train, X_test, y_train, y_test = preprocessing.load_data('./parkinsons_updrs.csv')

    # feature selection using random forest regressors
    X_train, X_test = preprocessing.random_forest_features(X_train, y_train, X_test)

    # feature selection using PCA
    # X_train, X_test = preprocessing.pca_features(X_train, X_test)

    # use SVR to predict UPDRS scores
    r_squared_svr, rmse_svr, mae_svr = svr.svr_model(X_train, y_train, X_test, y_test)
    print("R-squared value for SVR: " + str(round(r_squared_svr, 3)))
    print("RMSE for SVR: " + str(round(rmse_svr, 3)))

    print("MAE for SVR: " + str(round(mae_svr, 3)))

    # classify motor UPDRS scores and select optimal threshold
    X_train, X_test, y_train, y_test = preprocessing.load_data_for_classification('./parkinsons_updrs.csv')
    result = classifyMotor.find_threshold(X_train, X_test, y_train, y_test)
    print("Optimal threshold value: {}".format(result))


if __name__ == "__main__":
    main()
