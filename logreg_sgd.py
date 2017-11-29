# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:24:42 2017

@author: SamKao
"""
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt


def load_train_test_data(train_ratio=.8):
    data = pandas.read_csv('./HTRU_2.csv', header=None)
    X = data.drop(data.columns[8], axis = 1)
    X = numpy.concatenate((numpy.ones((len(X), 1)), X), axis = 1)
    y = data.values[:, 8].reshape(-1, 1)
    
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def gradient_descent(X, y, alpha = .001, iters = 1000, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    
    loss = numpy.array([])

    for i in range(iters):
        #loss = 0
        for index in range(n):
            z = -(numpy.dot(X[index].reshape(1, 9), theta))
            exp128 = numpy.exp(z)
            #print(exp128)
            y_hat = (1/(1 + exp128))
            #print(y_hat.shape)
            #loss = numpy.concatenate((loss, (y[index].reshape(1, 1)*numpy.log10(y_hat) + (1 - y[index].reshape(1, 1))*numpy.log10(1 - y_hat))), axis = 1)
            l = y[index].reshape(1, 1)*numpy.log10(y_hat) + (1 - y[index].reshape(1, 1))*numpy.log10(1 - y_hat)
            
            #print(loss[index])
            #print(loss.shape)
            theta = theta + alpha * (numpy.dot( X[index].reshape(d, 1), (y[index].reshape(1, 1) - y_hat)) - 0.001)
        loss = numpy.append(loss , (l / n))
        #print(loss / n)
    
    plt.plot(loss.reshape(-1, 1))
    plt.show()
    return theta


def predict(X, theta):
    return 1 / (1 + numpy.exp(-1*numpy.dot(X, theta)))


def main():
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    theta = gradient_descent(X_train_scale, y_train)
    
    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))


main()