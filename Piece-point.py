#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:52:26 2019

@author: prapti
"""

import cv2
import math

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
cv2.imshow('input image',img)

'''
PIECE WISE LINEAR OPERATION
'''

#(s1, r1) = (80, 50)
#(s2, r2) = (180, 150)

s1 = float(input("Enter s1:"))
r1 = float(input("Enter r1:"))
s2 = float(input("Enter s2:"))
r2 = float(input("Enter r2:"))

m1 = r1/s1

m2 = (r2-r1) / (s2-s1)

m3 = (255-r2) / (255-s2)

img = img.copy()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        intensity = img[i, j]
        if intensity <=s1:
            img[i, j] = intensity * m1
        elif intensity <= s2:
            img[i, j] = (m2 * (intensity-s1)) + r1
        else:
            img[i, j] = (m3 * (intensity-s2)) + r2
            
cv2.imshow('output image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
