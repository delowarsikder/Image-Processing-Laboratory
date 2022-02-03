# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:58:02 2020

@author: DelowaR
"""

import cv2
import  numpy as np
import math

img =cv2.imread('../images/apple.jpg',cv2.IMREAD_GRAYSCALE)
imgCopy=img.copy()
cv2.imshow('inputImage',img)

row,col=img.shape

mid=-2
for i in range(row):
    for j in range(col):
        px=img[i,j]
        x=(12/255)*px-6
        y=(math.exp(x-mid)-math.exp(-x+mid))/(math.exp(x-mid)+math.exp(-x+mid))#function tanh
        s=((y+1)/2)*255
        imgCopy.itemset((i,j),s)
        
cv2.imshow('output',imgCopy)                
        
cv2.waitKey(0)
cv2.destroyAllWindows()