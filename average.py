#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:28:57 2019

@author: prapti
"""


import cv2
import math
import numpy as np



image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

kernel = (np.array([[1, 1, 1], 
                    [1, 1, 1], 
                     [1, 1, 1]])) 

kernel_sum = kernel.sum()

i_width, i_height = image.shape[0], image.shape[1] 
k_width, k_height = kernel.shape[0], kernel.shape[1] 

filtered = np.zeros_like(image)

for y in range(i_height): 
    for x in range(i_width):
        weighted_pixel_sum = 0 
        for ky in range(-(k_height / 2), k_height - 1): 
            for kx in range(-(k_width / 2), k_width - 1):
                pixel = 0
                pixel_y = y - ky
                pixel_x = x - kx
 
                # boundary check: all values outside the image are treated as zero. 
                 # This is a definition and implementation dependent, it's not a property of the convolution itself. 
                 if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                     pixel = image[pixel_y, pixel_x]
                weight = kernel[ky + (k_height / 2), kx + (k_width / 2)] 
                weighted_pixel_sum += pixel * weight 
                filtered[y, x] = weighted_pixel_sum / kernel_sum 

cv2.imshow('DIY convolution', filtered) 

cv2.waitKey(0)
cv2.destroyAllWindows()