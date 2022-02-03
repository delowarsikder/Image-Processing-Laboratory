import cv2
import math
import numpy as np

img=cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

temp=np.zeros((img.shape[0]+2,img.shape[1]+2),dtype='uint8')
temp1=np.zeros((img.shape[0]+2,img.shape[1]+2),dtype='int8')
#

k=np.array([[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        temp.itemset((i+1,j+1),img.item(i,j))


for i in range(0,img.shape[0]-2):
    for j in range(0,img.shape[1]-2):
        intensity=temp1[i,j]
        #normalized=int(255*((intensity-mini)/(maxm-mini)))
        temp.itemset((i,j),intensity)

img=temp[1:img.shape[0]-1,1:img.shape[1]-1]
cv2.imshow('output',img)
cv2.waitKey(0)
cv2.destroyAllWindows()