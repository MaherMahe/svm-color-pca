from feature_extract import hog, color, fire
from feature_extract import *
import cv2
import numpy
import numpy as np
import image
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation, metrics  
from sklearn.externals import joblib


def pca(X):
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:
        M = dot(X, X.T)
        e, EV = linalg.eigh(M)
        tmp = dot(X.T, EV).T
        V = tmp[::-1]
        S = sqrt(e)[::-1]
    for i in range(V.shape[1]):
        V[:, i] /= S
    else:
        U, S, V = linalg.svd(X)
        V = V[:num_data]
    return V, S, mean_X

def training():

    pre = './image/'
    obj = ['fire','forest']
    tset = []
    tlabel = []
    for o in obj:
        path = pre+o+"/"
        listing = os.listdir(path)
        for file in listing:
            print(path + file)
            img = cv2.imread(path + file)
            if img.size:
                h = fire(img).flatten()
                tset.append(h)
                tlabel.append(o)

    Data = [tset, tlabel]
    X = Data[0]
    y = Data[1]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=0)
    pca = PCA(n_components=100)  # adjust yourself
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    clf = SVC()
    clf.fit(X_t_train, y_train)
    joblib.dump(pca, "./model/fm_pca.m")
    joblib.dump(clf, "./model/fm_clf.m")
    predict = clf.predict(X_t_test)
    ac_score = metrics.accuracy_score(y_test, predict)  
    cl_report = metrics.classification_report(y_test, predict)  
    print(ac_score)  
    print(cl_report)  
    print 'score', clf.score(X_t_test, y_test)
    print 'pred label', predict


def main():
    # training_shape()
    training()


if __name__ == '__main__':
    main()