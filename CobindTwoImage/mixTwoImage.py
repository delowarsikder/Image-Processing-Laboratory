import numpy
import cv2
import math

#img = cv2.imread('kuet.jpg', cv2.IMREAD_GRAYSCALE)

appleImage=cv2.imread('apple.jpg', 0)
orangeImage=cv2.imread('orange.jpg', 0)


appleImageCopy=appleImage.copy()
orangeImageCopy=orangeImage.copy()
mixtureImage=appleImage.copy()


## apple add first then orange
appleRow,appleCol=appleImageCopy.shape
w=appleCol/2
wide=20

for i in range(appleRow):
    f=1/40
    e=40
    for j in range(appleCol):
        if (j>w+wide):
            mixtureImage[i,j]=orangeImage[i,j]
        if (j>=w-wide and j<=w+wide):
              a=e*f
              b=1-a
              e=e-1
              x=a*appleImageCopy[i,j]
              y=b*orangeImageCopy[i,j]
              mixtureImage[i,j]=x+y
            
cv2.imshow('Apple', appleImage)
cv2.imshow('orange', orangeImage)

cv2.imshow('mixedImage : ', mixtureImage)

cv2.waitKey(0)     
cv2.destroyAllWindows();
