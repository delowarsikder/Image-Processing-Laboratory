import numpy as np
from math import *
import cv2
import copy
import matplotlib.pyplot as plt


def linerTransformation(img,row,col):
    copyImg=copy.deepcopy(img)
    for i in range(row):
        for j in range(col):
            temp=copyImg.item(i,j) ##extract a pixel from i,j posion
            if(temp>127):
                copyImg.itemset((i,j),255) ##set value in i,j position
            else:
                copyImg.itemset((i,j),0)
    return copyImg 


def negativeTransformation(img,rol,col):
    copyImg=copy.deepcopy(img)
    L=img.max()
    copyImg = np.array(L-img, dtype = 'uint8')
    # for i in range(row):
    #     for j in range(col):
    #         r=img[i,j]
    #         copyImg[i,j]=(L-r)
    return copyImg

def logTransformation(img,row,col):
    copyImg=copy.deepcopy(img)
    c=255/log(1 + img.max()) #max value 255 and max bit require to represent 
    for i in range(row):
        for j in range(col):
            r=img[i,j]
            copyImg[i,j]=c*log(r+1) ##log(256) max value is 8 , 
    return copyImg


def inverseLogTransformation(img,row,col):
    copyImg=copy.deepcopy(img)
    c=255/log(1 + img.max()) #max value 255 and max bit require to represent 
    for i in range(row):
        for j in range(col):
            r=img[i,j]
            copyImg[i,j]=2**(r/c)-1
    return copyImg


def gammaTransformation(img,row,col): ##power transform or gamma correction
    copyImg=copy.deepcopy(img)
    c=img.max()
    gamma=2
    for i in range(row):
        for j in range(col):
            r=img[i,j]
            copyImg[i,j]=c*(r/c)**gamma
    return copyImg

##start here

inputImg=cv2.imread("einstain.jpg",cv2.IMREAD_GRAYSCALE)
copyInputImage=copy.deepcopy(inputImg)
# convert log with manually
# cv2.imshow("Input Image: ",inputImg)
row,col=inputImg.shape
# print("max: ",inputImg.max())
# print("row: ",row)
# print("col: ",col)

##Linear Transform Image
linerTransformImage=linerTransformation(copyInputImage, row, col)
cv2.imshow("Liner Transform Output ",linerTransformImage)
cv2.imwrite("Liner_Transform_Output.png ",linerTransformImage)

###Negative image Transform
negativeTransformImage=negativeTransformation(copyInputImage, row, col)
cv2.imshow("Negative Transformation ",negativeTransformImage)
cv2.imwrite("NegativeTransformationOutput.png",negativeTransformImage)

###Log transformation 
logTransformationImage=logTransformation(copyInputImage, row, col)
cv2.imshow("Log Transformation",logTransformationImage)
cv2.imwrite("LogTransformationOutput.png",logTransformationImage)

###inverse Log transformation 
inverseLogTransformationImage=inverseLogTransformation(copyInputImage, row, col)
cv2.imshow("Inverse Log Transformation",inverseLogTransformationImage)
cv2.imwrite("InvereseLogTransformationOutput.png",inverseLogTransformationImage)

###power transformation  or gamma correction
gammaTransformationImage=gammaTransformation(copyInputImage, row, col)
cv2.imshow("gamma Transformation",gammaTransformationImage)
cv2.imwrite("GammaTransformationOutput.png",gammaTransformationImage)

####subplot images
#Change the figure size
plt.figure(figsize=[11, 8])


plt.subplot(231)
plt.imshow(copyInputImage,cmap='gray')
plt.title("input Images"),plt.xticks([]),plt.yticks([])

plt.subplot(232)
plt.imshow(linerTransformImage,cmap='gray')
plt.title("linerTransformImage"),plt.xticks([]),plt.yticks([])

plt.subplot(233)
plt.imshow(logTransformationImage,cmap='gray')
plt.title("logTransformationImage"),plt.xticks([]),plt.yticks([])

plt.subplot(234)
plt.imshow(inverseLogTransformationImage,cmap='gray')
plt.title("inverseLogTransformationImage"),plt.xticks([]),plt.yticks([])

plt.subplot(235)
plt.imshow(negativeTransformImage,cmap='gray')
plt.title("negativeTransformImage"),plt.xticks([]),plt.yticks([])

plt.subplot(236)
plt.imshow(gammaTransformationImage,cmap='gray')
plt.title("gammaTransformationImage")



plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()