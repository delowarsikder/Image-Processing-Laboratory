import numpy as np
import cv2
from math import *
from copy import deepcopy
import matplotlib.pyplot as plt
inputImage=cv2.imread("erosion_dialationInput.png",cv2.IMREAD_GRAYSCALE)
copyImage=deepcopy(inputImage)

##find binay threshold
threshold,binaryImage=cv2.threshold(copyImage,127,255,cv2.THRESH_BINARY)
#filtersize 5x5
filterK=np.ones((5,5))
#shape of binary image and kernel
S=binaryImage.shape
F=filterK.shape

#convert into 0 and 1
binaryImage=binaryImage/255
dilationImage=deepcopy(binaryImage)
erosionImage=deepcopy(binaryImage)

#add padding on image
#padding size calculation
R=S[0]+F[0]-1
C=S[1]+F[1]-1

##create new blank image with R and C
N=np.zeros((R,C))

##insert input into blank image
for i in range(S[0]):
    for j in range(S[1]):
        N[i+F[0]//2,j+F[1]//2]=binaryImage[i,j]

###apply dilation

for i in range(S[0]):
    for j in range(S[1]):
        k=N[i:i+F[0],j:j+F[1]]
        result=(k==filterK)
        if(result.any()):
           dilationImage[i,j]=1 
        else:
           dilationImage[i,j]=0

##apply erosion
for i in range(S[0]):
    for j in range(S[1]):
        k=N[i:i+F[0],j:j+F[1]]
        result=(k==filterK)
        final=np.all(result==True)
        if(final):
           erosionImage[i,j]=1 
        else:
           erosionImage[i,j]=0


cv2.imshow("Dilation ",dilationImage)
cv2.imshow("Erosion ",erosionImage)

plt.subplot(221)
plt.imshow(copyImage,cmap='gray')
plt.title("Input Image")

plt.subplot(222)
plt.imshow(binaryImage,cmap='gray')
plt.title("Binary Image")

plt.subplot(223)
plt.imshow(N,cmap='gray')
plt.title("Padding Image")

plt.show()

boundary=dilationImage-binaryImage

cv2.imshow("boundary",boundary)

# cv2.imshow("Input Image",copyImage) # 

cv2.waitKey(0)
cv2.destroyAllWindows()