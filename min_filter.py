#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 02:48:10 2019

@author: prapti
"""

import cv2
import numpy as np 

img = cv2.imread('noise.png',  cv2.IMREAD_GRAYSCALE)
new_img = img.copy()
prop = img.shape

k=3
#we take a window and find the median of intensity values 
#in that window and assigned it to the current pixel

############### 3x3 kernel ############### 
for i in range(k//2, prop[0]-1-k//2):
    for j in range(k//2, prop[0]-1-k//2):
        
        min = 255
        for x in range(-k//2, (k//2)+1):
            for y in range(-k//2, (k//2)+1):
                a = img.item(i+x, j+y)
                if a < min:
                    min = a
        b = min
        new_img.itemset((i,j), b)
q      
       
cv2.imshow('original', img)
cv2.imshow('output', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
