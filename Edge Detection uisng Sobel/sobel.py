import numpy as np
import cv2

#import input image
image_original = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
#convert to gray color
image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
#show input
cv2.imshow("input",image_gray)

#get image width and height
[rows, columns] = np.shape(image_gray)
#create output image only with zero same size as input image
sobel_filtered_image = np.zeros(shape=(rows, columns))
#same for x derivative and y derivative
sobel_filtered_x = np.zeros(shape=(rows, columns))
sobel_filtered_y = np.zeros(shape=(rows, columns))

#sobel kernel
sobel_y = np.array([[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]])

sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])


#iterate through the image

for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(sobel_x, image_gray[i:i + 3, j:j + 3]))     #get x derivative
        sobel_filtered_x[i+1,j+1]=gx                                        #store x derivative
        gy = np.sum(np.multiply(sobel_y, image_gray[i:i + 3, j:j + 3]))     #get y derivative
        sobel_filtered_y[i+1,j+1]=gy                                        #store y derivative
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)     #find the megnitude

#normalize / scale the image 0 to 255
cv2.normalize(sobel_filtered_x, sobel_filtered_x, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobel_filtered_y, sobel_filtered_y, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobel_filtered_image, sobel_filtered_image, 0, 255, cv2.NORM_MINMAX)
        
#round up and type cast to int from float
sobel_filtered_x = np.round(sobel_filtered_x).astype(np.uint8)
sobel_filtered_y = np.round(sobel_filtered_y).astype(np.uint8)
sobel_filtered_image = np.round(sobel_filtered_image).astype(np.uint8)

#output x-derivative and y-derivatiive
cv2.imshow("sobel x",sobel_filtered_x)
cv2.imshow("sobel y",sobel_filtered_y)

#use a threshold value 
ret1,th1 = cv2.threshold(sobel_filtered_image,100,255,cv2.THRESH_BINARY)
#out[ut final image]
cv2.imshow("sobel",sobel_filtered_image)
        
cv2.waitKey(0)
cv2.destroyAllWindows()