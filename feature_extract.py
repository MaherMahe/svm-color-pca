import cv2
import numpy
import numpy as np
import os
import scipy.linalg as linA
import glob
from scipy import *
from scipy import signal

def fire(img):
    rth = 115
    sth = 55
    height,width,t = img.shape
    b, g, r = cv2.split(img)
    fireImg = b.copy()
    white = 0
    for i in range(height):
        for j in range(width):
            bi = img[i,j,0]
            gi = img[i,j,1]
            ri = img[i,j,2]
            maxValue = max(max(bi, gi), ri)
            minValue = min(min(bi, gi), ri)
            s = 1 - 3.0*minValue/sum([ri,gi,bi])
            v = ((255 - ri) * sth / rth)
            if (ri > rth and ri >= gi and gi >= bi and s > 0.20 and s > v):
                fireImg[i,j]=255
                white+=1
            else:
                fireImg[i,j]=0.0
    return fireImg

def fire2(img):

    rth = 115 #115~135
    sth = 55 #55~65
    height,width,t = img.shape
    b, g, r = cv2.split(img)
    fireImg = img.copy()
    white = 0
    for i in range(height):
        for j in range(width):
            bi = img[i,j,0]
            gi = img[i,j,1]
            ri = img[i,j,2]
            maxValue = max(max(bi, gi), ri)
            minValue = min(min(bi, gi), ri)
            s = 1 - 3.0*minValue/sum([ri,gi,bi])
            v = ((255 - ri) * sth / rth)
            if (ri > rth and ri >= gi and gi >= bi and s > 0.20 and s > v):
                fireImg[i,j,0]=bi
                fireImg[i,j,1]=gi
                fireImg[i,j,2]=ri
                white+=1
            else:
                fireImg[i,j,0]=0.0
                fireImg[i,j,1]=0.0
                fireImg[i,j,2]=0.0
    return fireImg


def hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist.flatten()

def color(img):
    b, g, r = cv2.split(img)
    v1 = cv2.calcHist([r], [0], None, [256], [0.0,255.0])
    v2 = cv2.calcHist([g], [0], None, [256], [0.0,255.0])
    v3 = cv2.calcHist([b], [0], None, [256], [0.0,255.0])
    #v = v.flatten()
    #hist = v / sum(v)
    return np.concatenate((v1,v2,v3),axis=0).flatten()



def corner(img):

    rth = 115 #115~135
    sth = 30 #55~65
    height,width,t = img.shape
    b, g, r = cv2.split(img)
    fireImg = b.copy()
    resImg = g.copy()
    white = 0
    for i in range(height):
        for j in range(width):
            bi = img[i,j,0]
            gi = img[i,j,1]
            ri = img[i,j,2]
            resImg[i,j] = 0.0
            maxValue = max(max(bi, gi), ri)
            minValue = min(min(bi, gi), ri)
            s = 1 - 3.0*minValue/sum([ri,gi,bi])
            v = ((255 - ri) * sth / rth)
            if (ri > rth and ri >= gi and gi >= bi and s > 0.20 and s > v):
                fireImg[i,j]=255.0
                white+=1
            else:
                fireImg[i,j]=0.0
    corners = cv2.goodFeaturesToTrack(fireImg, 100, 0.01, 80)
    corners = np.int0(corners)
    for corneri in corners:
        x,y = corneri.ravel()
        resImg[y,x]=255.0
    print(corners.size)
    return [corners.size,1]