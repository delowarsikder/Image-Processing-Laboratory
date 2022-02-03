# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 01:53:08 2020

@author: DelowaR
"""

import cv2
import numpy as np
import math

img=cv2.imread('../images/apple.jpg',cv2.IMREAD_GRAYSCALE)  
out1=img.copy()
cv2.imshow('inputImage',out1)

def contrast(px,r1,s1,r2,s2):
    if (0<px and px<=r1):
        return (r1/s1)*px
    elif (px<r2 and r1<px):
        return ((s2-s1)/(r2-r1)*(px-r1))+s1
    else:
        return ((255-s1)/(255-r1)*(px-r1))+s1



#define parameter
r1=25
s1=20
r2=150
s2=100
c=img.max()
row,col=out1.shape
for i in range(row):
    for j in range(col):
      px=img.item(i,j)
      s=contrast(px,r1,s1,r2,s2)
      out1.itemset((i,j),s)
cv2.imshow('ContrastStreching',out1)
cv2.waitKey(0)
cv2.destroyAllWindows()