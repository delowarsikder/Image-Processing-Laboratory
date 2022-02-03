#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 07:03:15 2019

@author: prapti
"""

import cv2
import math

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
cv2.imshow('input image',img)

'''
 GAMMA TRANSFORMATION
'''

gamma = float(input("Enter Gamma Value:"))
#gamma = 1.5

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        intensity = img[i, j]
        img[i,j] = 255 * math.pow((intensity/255), gamma)


cv2.imshow('output image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ref-  https://stackoverflow.com/questions/16521003/gamma-correction-formula-gamma-or-1-gamma