import numpy as np
import cv2
import math
from copy import deepcopy

# to normalize the image

def normalize(img):
    normal_image = np.zeros(img.shape)
    
    max_img = img.max()
    min_img = img.min()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            normal_image[i][j]= (img[i][j] - min_img)/ (max_img - min_img) *255
            
    return np.array(normal_image, dtype = 'uint8')


# noise reduce using guassian smoothing
def guassian_smoothing(input_image):
    guass_kernel = np.array([[1,2,1],
                             [2,4,2],
                             [1,2,1]])
    
    smooth_image = cv2.filter2D(input_image, -1, guass_kernel)
    smooth_image_normalized = normalize(smooth_image.copy())
    
    return smooth_image, smooth_image_normalized

# edge detection operator to calculate the changes correspond to the edge
def edge_detection_operator_sobel(input_image2):
    sobel_x = np.array([[-1,0,1],
                              [-2,0,2],
                              [-1,0,1]])
    sobel_y = np.array([[1,2,1],
                             [0,0,0],
                             [-1,-2,-1]])
    
    sobel_x_img = cv2.filter2D(input_image2, -1, sobel_x)
    sobel_x_img_normalized = normalize(sobel_x_img.copy())
    
    sobel_y_img = cv2.filter2D(input_image2, -1, sobel_y)
    sobel_y_img_normalized = normalize(sobel_y_img.copy())
    
    sobel_image = np.sqrt((sobel_x_img**2) + (sobel_y_img**2))
    sobel_image_normalized = normalize(sobel_image.copy())
    
    theta = np.arctan2(sobel_y_img,sobel_x_img)
    
    return sobel_x_img_normalized, sobel_y_img_normalized, sobel_image, sobel_image_normalized, theta
    

# non maximum supression
def non_max_supression(input_image, theta):
    row, col = input_image.shape
    supressed_image = deepcopy(input_image)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1,row-1):
        for j in range(1,col-1):
            try:
                n1 = 255
                n2 = 255
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    n1 = input_image[i, j+1]
                    n2 = input_image[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    n1 = input_image[i+1, j-1]
                    n2 = input_image[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    n1 = input_image[i+1, j]
                    n2 = input_image[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    n1 = input_image[i-1, j-1]
                    n2 = input_image[i+1, j+1]

                if (input_image[i,j] >= n1) and (input_image[i,j] >= n2):
                   supressed_image[i,j] = input_image[i,j]
                else:
                    supressed_image[i,j] = 0

            except IndexError as e:
                pass
    
    supressed_image_normalized = normalize(supressed_image.copy())
    return supressed_image, supressed_image_normalized
    

#Double threshold
#def double_threshold(input_image, lowThresholdRatio=0.3, highThresholdRatio=0.15):
def double_threshold(in_img, lowThresholdRatio=0.3, highThresholdRatio=0.15):
    
    highThreshold = int(in_img.max() * highThresholdRatio);
    lowThreshold = int(highThreshold * lowThresholdRatio);
    
    Thressed_image = deepcopy(in_img)
    
    weak = 100
    strong = 255
    
    Thressed_image[in_img >= highThreshold] = strong
    Thressed_image[in_img < lowThreshold] = 0
    
    weak_i, weak_j = np.where((in_img <= highThreshold) & (in_img >= lowThreshold))
    
    Thressed_image[weak_i, weak_j] = weak
    
    return Thressed_image

#Edge Tracking by Hysteresis
def hysteresis(img, weak=100, strong=255):
    
    row, col = img.shape  
    for i in range(1, row-1):
        for j in range(1, col-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def canny_edge_detection_algo(input_image):
    # step-1: noise reduction
    noise_reduced_image, noise_reduced_image_normalized = guassian_smoothing(input_image)
    
    # step-2: gradient calculation
    sobel_x_img_normalized, sobel_y_img_normalized, sobel_image, sobel_image_normalized, theta = edge_detection_operator_sobel(noise_reduced_image.copy())
    
    # step-3: non-maximum supression
    supressed_image, supressed_image_normalized = non_max_supression(sobel_image_normalized.copy(), theta)
    
    # step-5: Double threshold
    threshold_image = double_threshold(supressed_image_normalized.copy())
    
    # step-6: Edge Tracking by Hysteresis
    edged_image = hysteresis(threshold_image.copy())
    
    return noise_reduced_image_normalized, sobel_x_img_normalized, sobel_y_img_normalized,sobel_image_normalized, supressed_image_normalized, threshold_image, edged_image
    
    

img_input=cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img = deepcopy(img_input)
img = np.array(img, dtype='float32')

img_noise_reduced, sobel_x_img_normalized, sobel_y_img_normalized, sobel_image_normalized, supressed_image_normalized, threshold_image, canny_edge_image = canny_edge_detection_algo(img)

cv2.imshow("Input Image", img_input)

cv2.imshow("normalize noise reduction", img_noise_reduced)

cv2.imshow("sobel_x_img_normalized", sobel_x_img_normalized)

cv2.imshow("sobel_y_img_normalized", sobel_y_img_normalized)

cv2.imshow("sobel_image_normalized", sobel_image_normalized)

cv2.imshow("Non-maximum supression",supressed_image_normalized)

cv2.imshow("Double_threshold_image", threshold_image)

cv2.imshow("canny_edge_detection_image", canny_edge_image)

cv2.waitKey(0)
cv2.destroyAllWindows()