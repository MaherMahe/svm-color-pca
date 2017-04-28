from feature_extract import hog,color,fire
import cv2
import numpy
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.externals import joblib

def main():
    img = cv2.imread('./source/1.jpg')
    cv2.resize(img,(400,300))
    cv2.imshow('fire',fire(img))
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()

if __name__ == '__main__':main()