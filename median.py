"""
Created on Wed Mar 13 02:29:26 2019

@author: prapti
"""

import cv2
import numpy as np 

img = cv2.imread('noise.png',  cv2.IMREAD_GRAYSCALE)
new_img = img.copy()
prop = img.shape

k = 3


for i in range(k//2, prop[0]-1-k//2):
    for j in range(k//2, prop[0]-1-k//2):
        
        roi = []
        for x in range(-k//2, (k//2)+1):
            for y in range(-k//2, (k//2)+1):
                roi.append( img[x+i][j+y] )
    
        roi.sort()
    
        new_img[i][j] = roi[len(roi)//2]


cv2.imshow('original', img)
cv2.imshow('output', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
