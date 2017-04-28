from feature_extract import hog,color,fire,corner
import cv2
import numpy
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.externals import joblib

def predict_color(path):

    training_set = []
    test_set=[]
    color_test_set=[]
    training_labels=[]
    result_list=[]
    ######     Now testing Color      ########################

    clf = joblib.load('./model/fm_clf.m')
    pca = joblib.load('./model/fm_pca.m')

    img = cv2.imread(path)
    i = fire(img).flatten().reshape(1,-1)
    i_t = pca.transform(i)
    res = clf.predict(i_t)
    return res

def main():
    total = 0
    correct = 0
    pre = "./image/test/"
    sample = ["forest","fire"]
    for s in sample:
        totals = 0
        corrects = 0
        path = pre+s+"/"
        listing=os.listdir(path)
        for i,filei in enumerate(listing):
            img=path+filei
            #print img
            #print predict_shape(img)
            totals+=1
            if predict_color(img)==s:
                print 'wow'
                corrects+=1
            else:
                print(path+filei)
                print 'oops'
        total+=totals
        correct+=corrects
        print (corrects,totals)
    print (correct,total)
    
if __name__ == '__main__':main()






