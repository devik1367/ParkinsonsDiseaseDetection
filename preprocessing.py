import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize


def load_data(path):
    parkinsons_data = pd.read_csv(path)

    X = parkinsons_data.drop(['total_UPDRS', 'motor_UPDRS'], axis=1)
    y = parkinsons_data.loc[:, 'total_UPDRS']

    # y2 is also a response variable but right now we are predicting only total_UPDRS score
    y2 = parkinsons_data.loc[:, 'motor_UPDRS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


# load data with motor UPDRS as response variable and with only vocal features
def load_data_for_classification(path):
    parkinsons_data = pd.read_csv(path)

    # use only vocal features for classification
    X = parkinsons_data.drop(['age', 'sex', 'test_time', 'subject#', 'total_UPDRS', 'motor_UPDRS'], axis=1)
    # response variable is motor UPDRS
    y = parkinsons_data.loc[:, 'motor_UPDRS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def random_forest_features(X_train, y_train, X_test):

    model = SelectFromModel(RandomForestRegressor(n_estimators=1000))
    model.fit(X_train, y_train)

    selected_feat = X_train.columns[(model.get_support())]

    print("Attributes selected are " + str(selected_feat))

    X_train = X_train[selected_feat]
    X_test = X_test[selected_feat]

    return X_train, X_test


def pca_features(X_train, X_test):
    scalar = StandardScaler()
    scalar.fit(X_train)

    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    pca = PCA(0.95)
    pca.fit(X_train)

    print("number of principal components selected are " + str(pca.n_components_))
    print(pca.components_)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    X_train = pd.DataFrame(data=X_train)
    X_test = pd.DataFrame(data=X_test)

    return X_train, X_test
