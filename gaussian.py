# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:k_size1:03 2019

@author: softeng
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('input', img)

k_size = int(input('Enter kernel size : '))
k = np.ones((k_size, k_size))/9

print(k)

for i in range(k_size, img.shape[0] - k_size):
    for j in range(k_size, img.shape[1] - k_size):
        sum = 0.0
        for x in range(k_size):
            for y in range(k_size):
                a = img.item(i+x, j+y)
                b = k[x][y] * a
                sum = sum + b
        img.itemset((i,j), sum)
        
cv2.imshow('output',img)
                
plt.imshow(k, cmap = 'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
