#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 02:45:06 2019

@author: prapti
"""


import cv2
import math

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
cv2.imshow('input image',img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j] = 255 - img[i, j]

cv2.imshow('output image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()