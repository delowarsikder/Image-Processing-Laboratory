import cv2
from skimage.exposure import rescale_intensity
import numpy as np
import math
from matplotlib import pyplot as plt
#from Util import *


#[x', y', 1].T = [R=[cos$, -sin$, 0], [sin$, cos$, 0], [0, 0, 1]]
# R.inv = [[cos$, sin$, 0], [-sin$, cos$, 0], [0, 0, 1]]

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (500, 500))
cv2.imshow('Original', img)



def rotate(img, deg):
    img_rotated = img.copy()
    for u in range(img.shape[0]):
        for v in range(img.shape[1]):
            x = round(u*math.cos(deg) + v*math.sin(deg))
            y = round(-u*math.sin(deg) + v*math.cos(deg))
            if x<0 or x>=img.shape[0] or y<0 or y>=img.shape[1]:
                img_rotated[u, v] = 0
            else:
                img_rotated[u, v] = img[x, y]
    return img_rotated

def rotate_center(img, deg, point):
    img_rotated = img.copy()
    for u in range(0, img.shape[0]-1):
        for v in range(0, img.shape[1]-1):
            
            x = round((u-point[0])*math.cos(deg) + (v-point[1])*math.sin(deg))
            y = round(-(u-point[0])*math.sin(deg) + (v-point[1])*math.cos(deg))
            #u = u+point[0]
            #v = v+point[1]
            x = x+point[0]
            y = y+point[1]
            if x<0 or x>=img.shape[0] or y<0 or y>=img.shape[1]:
                img_rotated[u, v] = 0
            else:
                img_rotated[u, v] = img[x, y]
    return img_rotated

img_rotated = rotate(img, math.pi/180*10)
x = int(input('Enter center X: '))
y = int(input('Enter center Y: '))
img_rotated_centered = rotate_center(img, math.pi/180*10, (x, y))
cv2.imshow('Rotated', img_rotated)
cv2.imshow('Rotated_centered', img_rotated_centered)

cv2.waitKey(0)
cv2.destroyAllWindows()