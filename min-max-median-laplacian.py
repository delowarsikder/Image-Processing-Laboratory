import cv2
import numpy as np

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

def max_img(roi):
    return roi.max()

def min_img(roi):
    return roi.min()

def median_img(roi):
    return np.median(roi)


def operation(image, kernel_size, func):
    iH, iW = image.shape[0], image.shape[1]
 
    pad = (kernel_size - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    
    output = np.zeros((iH, iW), dtype='uint8')
    
    for y in range(pad, iH + pad):
        for x in range(pad, iW + pad):

            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            
            output[y-pad, x-pad] = func(roi)
            
    return output

cv2.imshow('original', img)

img_min = operation(img, 5, min_

cv2.imshow('min', operation(img, 5, min_img))
cv2.imshow('max', operation(img, 5, max_img))
cv2.imshow('Median', operation(img, 5, median_img))

cv2.waitKey(0)
cv2.destroyAllWindows()
