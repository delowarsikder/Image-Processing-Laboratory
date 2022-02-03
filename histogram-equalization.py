#Imporitn library
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt



img=cv2.imread("hist.jpg",0)
cv2.imshow("input image",img)

#histogram of input iamge
plt.hist(img.ravel(),256,[0,255])
plt.show()


#equalization Function

row,col=img.shape[0:2]
size=row*col

val={}
s={}
for i in range(256):
    val[i]=0
    s[i]=0

#calculating nk (frequency)
for i in range(row):
    for j in range(col):
        m=img[i,j]
        val[m]=val[m]+1
        
#calculating probability and cdf
s=0
for i in range(256):
    val[i]=val[i]/size
    if i>0:
        val[i]=val[i]+val[i-1]

#calculating s[]
for i in range(256):
    val[i]=round(val[i]*255)

#setting output image   
for i in range(row):
    for j in range(col):
        m=img[i,j]
        x=val[m]
        img[i,j]=x
        
#histogram after equalization        
plt.hist(img.ravel(),256,[0,255])
plt.show()

cv2.imshow("Output image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()