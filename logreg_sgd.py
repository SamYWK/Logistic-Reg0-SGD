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


def s_gradient_descent(X, y, alpha = .001, iters = 100, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    
    #loss = numpy.array([])

    for i in range(iters):
        for index in range(n):
            z = -(numpy.dot(X[index].reshape(1, 9), theta))
            exp128 = numpy.exp(z)
            y_hat = (1/(1 + exp128))
            #l = y[index].reshape(1, 1)*numpy.log10(y_hat) + (1 - y[index].reshape(1, 1))*numpy.log10(1 - y_hat)
            theta = theta + alpha * (numpy.dot( X[index].reshape(d, 1), (y[index].reshape(1, 1) - y_hat)) - 0.001)
        #loss = numpy.append(loss , (l / n))
    '''
    plt.figure(1)
    plt.plot(loss.reshape(-1, 1))
    plt.show()'''
    return theta


def predict(X, theta):
    return 1 / (1 + numpy.exp(-1*numpy.dot(X, theta)))


def plot_ROC_curve(y_test, y_hat):
    ys = numpy.concatenate((y_test, y_hat), axis = 1)
    ys_df = pandas.DataFrame(data = ys, columns = ['y_test', 'y_hat'])
    ys_df_sort = (ys_df.sort_values(by=['y_hat'])).reset_index(drop =True)
    
    TPR_array = numpy.array([])
    FPR_array = numpy.array([])

    for threshold in range(0, y_test.size):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        TPR = 0
        FPR = 0 
        
        for index in range(threshold):
            if(ys_df_sort.values[index, 0] == 0):
                TN += 1
            elif(ys_df_sort.values[index, 0] == 1):
                FN += 1
        for index in range(threshold + 1, y_test.size):
            if(ys_df_sort.values[index, 0] == 0):
                FP += 1
            elif(ys_df_sort.values[index, 0] == 1):
                TP += 1
        
        if ((TP + FN) == 0):
            TPR = 0
        elif ((FP + TN) == 0):
            FPR = 0
        else:    
            TPR = TP/(TP + FN)
            FPR = FP/(FP + TN)
        print(TP, FP, TN, FN)
        #print(TPR, FPR)
        TPR_array = numpy.append(TPR_array, TPR)
        FPR_array = numpy.append(FPR_array, FPR)
    #print(TPR_array, FPR_array)
    #print(TPR_array)
    plt.figure(2)
    #plt.xlim(xmax = 1, xmin = 0)
    plt.xlabel('FPR')
    #plt.ylim(ymax = 1, ymin = 0)
    plt.ylabel('TPR')
    plt.plot(FPR_array, TPR_array)
    plt.show()
    return None

def main():
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    theta = s_gradient_descent(X_train_scale, y_train)
    
    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))
    
    plot_ROC_curve(y_test, y_hat)

main()