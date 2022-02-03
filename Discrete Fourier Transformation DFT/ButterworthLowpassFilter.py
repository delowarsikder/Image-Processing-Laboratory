import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img2.jpg',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

mag = np.abs(fshift)
magnitude_spectrum = 20*np.log(mag)

#def Gaussian_filter(m,n,sigma=50):
def BLF(m, n, d0, n0):
    
    u = int(m/2)
    v = int(n/2)
    
    G = np.zeros((m,n), dtype=float)
    
    
# =============================================================================
#     c = 1/(2*np.pi*(sigma**2))
#     c2 = 2*(sigma**2)
# =============================================================================
    
    for i in range(-u,u):
        for j in range(-v,v):
            #G[i+u,j+v] = c*np.exp(-((i**2)+(j**2))/c2)
            if( i==0 and j==0):
                continue
            Duv = np.floor((i**2 + j**2)**0.5)
            G[i+u,j+v] = 1/(1 + (d0/Duv))**(2*n0)

    
    #print(G)
    
    
    #G = np.fft.ifftshift(G)

    return G
    

#mag, ang = cv2.cartToPolar(fshift[:,:,0],fshift[:,:,1])


def normalize(img):
    img = ((img-img.min())/( img.max()-img.min()))*255.0
    return img.astype(np.uint8)


m, n = img.shape
print(img.shape)
sigma = int(input())
#G = Gaussian_filter(m, n, 20)
G = BLF(m, n, sigma, 0.6)

gg=normalize(G)
cv2.imshow("Filter",gg)
print(G)
ang  = np.angle(fshift)
#combined = fshift * G 
mag = mag * G
combined = np.multiply(mag,np.exp(1j*ang))



#imgCombined = np.real(np.fft.ifft2(np.fft.ifftshift(combined)))
imgCombined = np.real(np.fft.ifft2(np.fft.ifftshift(combined)))
cv2.imshow("input image",img)
cv2.imshow("output image",imgCombined)
plt.subplot(121),
plt.imshow(img,cmap='gray')
plt.title("Input image"), plt.xticks([]),plt.yticks([])
plt.subplot(122),
plt.imshow(magnitude_spectrum,cmap='gray' )
plt.title("Magnitude Spectrum"),  plt.xticks([]),plt.yticks([])
plt.show()

plt.subplot(121),
plt.imshow(imgCombined,cmap='gray')
plt.title("output image"), plt.xticks([]),plt.yticks([])
plt.subplot(122),
plt.imshow(normalize(G),cmap='gray' )
plt.title("Gaussian Spectrum"),  plt.xticks([]),plt.yticks([])

#plt.imshow(img,cmap='gray')
#plt.imshow(imgCombined,cmap='gray')


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()