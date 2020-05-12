import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import preprocessing

learning_rate=0.0001
epochs=200
MAE_val=[]

def forward_propogation(X_input,weights,b):
    return (b+np.dot(weights,X_input))

def cost(z_predicted,y_actual):
    return ((float(1)/(float(2)*(y_actual.shape[1])))*np.sum(np.square(z_predicted-y_actual)))

def back_propogation(X,y,z):
    dz=(float(1)/float(y.shape[1]))*(z-y)
    return (np.dot(dz,X.T)), np.sum(dz)

def gradient_descent(w,b,dw,db,learning_rate):
    return (w-learning_rate*dw), (b-learning_rate*db)

def linear_regression(X_train,y_train,X_test,y_test,learning_rate,epochs):
    length_w=X_train.shape[0]
    w,b=np.random.randn(1, length_w),0
    rows_training,rows_validation=y_train.shape[1],y_test.shape[1]

    for i in range(epochs):
        z_train=forward_propogation(X_train, w, b)
        training_cost=cost(z_train, y_train)
        dw, db=back_propogation(X_train, y_train, z_train)
        w,b=gradient_descent(w,b,dw,db,learning_rate)
        training_MAE=(float(1)/float(rows_training))*np.sum(np.abs(z_train-y_train))

        z_test=forward_propogation(X_test, w, b)
        validation_cost=cost(z_test, y_test)  
        validation_MAE=(float(1)/float(rows_validation))*np.sum(np.abs(z_test-y_test))
        MAE_val.append(validation_MAE)

    return training_MAE, validation_MAE, np.sqrt(((z_test - y_test) ** 2).mean())

X_train, X_test, y_train, y_test = preprocessing.load_data('./parkinsons_updrs.csv')
X_train, X_test = preprocessing.random_forest_features(X_train, y_train, X_test)
#X_train, X_test = preprocessing.pca_features(X_train, X_test)
X_train, X_test=X_train.T, X_test.T
y_train, y_test=np.array([y_train]), np.array([y_test])

training_MAE, validation_MAE, RMSE=linear_regression(X_train, y_train, X_test, y_test, learning_rate, epochs)

print ("learning_rate "+str(learning_rate)+" epochs "+str(epochs))
print("Training MAE")
print(training_MAE)
print("Validation MAE")
print(validation_MAE)
print ("RMSE")
print (RMSE)


linear_regression=linear_model.LinearRegression()
model=linear_regression.fit(X_train.T, y_train.T)
predictions=linear_regression.predict(X_test.T)

print ("MAE with the Sklearn library")
MAE_with_library=(1.0/y_test.shape[1])*np.sum(np.abs(predictions-y_test.T))
print(MAE_with_library)
print ("RMSE with the SKlearn library")
RMSE_with_library=(1.0/y_test.shape[1])*np.sum(np.abs(predictions-y_test.T))
print(np.sqrt(((predictions - y_test) ** 2).mean()))


plt.plot(MAE_val)
plt.xlabel("Iterations")
plt.ylabel("MAE values")
plt.show()
