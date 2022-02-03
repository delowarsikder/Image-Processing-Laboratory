import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread('hist.jpg',0)
cv2.imshow("input",img)
print(img.shape)
plt.hist(img.ravel(),256, [0,255])
plt.show()

m1=float(input("Mean1= "))
sd1=float(input("SD1= "))
m2=float(input("Mean2= "))
sd2=float(input("SD2= "))
p=math.pi
t=math.sqrt(2*p)

def calc(x,m,sd):
    a=(x-m)/sd
    temp=(1/(sd*t))*(math.exp((-0.5)*(a*a)))
    return temp

g1=[]
g2=[]
g=[]
summ=0

cnt1=np.zeros((256,),np.int32)
sk1=np.zeros((256,),np.int32)
sk2=np.zeros((256,),np.int32)
sk=np.zeros((256,),np.int32)

h=img.shape[0]
w=img.shape[1]

for x in range(256):
    t1=calc(x,m1,sd1)
    t2=calc(x,m2,sd2)
    g1.append(t1)
    g2.append(t2)
    g.append(t1+t2)
    summ=summ+t1+t2
    
for x in range(256):
    g[x]=g[x]/summ    

plt.plot(np.array(range(256)),g1) 
plt.plot(np.array(range(256)),g2) 
plt.plot(np.array(range(256)),g) 
plt.title("Gaussian()") 
plt.xlabel("X") 
plt.ylabel("Y") 
plt.show() 

p=0    
for x in range(256):
    p= (g[x]) + p
    s= 255*p
    s= round(s)
    sk2[x]=s     
    
for i in range(h):
    for j in range(w):
        temp=img[i][j]
        cnt1[temp]= cnt1[temp]+1
        
p=0
            
for x in range(len(cnt1)):
    p= (cnt1[x]/(h*w)) +p
    s= 255 * p
    s= round(s)
    sk1[x]=s
    
       
for x in range(len(sk1)):
    temp=sk1[x]
    rep=0
    dif=100000000
    for y in range(len(sk2)):
        t=abs(temp-sk2[y])
        if t<dif:
            dif=t
            rep=y
    sk[x]=rep
        
plt.plot(np.array(range(256)),sk2) 
plt.plot(np.array(range(256)),sk1) 
plt.plot(np.array(range(256)),sk) 
plt.title("Equlization") 
plt.xlabel("X") 
plt.ylabel("Y") 
plt.show() 


plt.hist(sk2.ravel(),256, [0,255])
plt.show()
plt.hist(sk1.ravel(),256, [0,255])
plt.show()
plt.hist(sk.ravel(),256, [0,255])
plt.show()

#plt.hist(sk2.ravel(),256, [0,255])
#plt.hist(sk1.ravel(),256, [0,255])
#plt.hist(sk.ravel(),256, [0,255])
#plt.show()


img2=np.ones((400,600),img.dtype)
for i in range(h):
    for j in range(w):
        temp=img.item(i,j)
        img2[i][j]=sk[temp]
        
plt.hist(img2.ravel(),256, [0,255])
plt.show()
        
plt.hist(img.ravel(),256,[0,255])
plt.hist(img2.ravel(),256, [0,255])
plt.show()

cv2.imshow("output",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()