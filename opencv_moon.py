#making conv filter (filter, padding, stride(need default) )
#height:행 , width:열 chennel:color층
import cv2
import numpy as np
import matplotlib.pyplot as plt
file = 'moon'#type your file name
image = cv2.imread('C:/Users/dnfld/Desktop/deeplearning_learning/image/'+file+'.PNG',cv2.IMREAD_GRAYSCALE)
height, width = image.shape
cd = 10 #cognition default
conv = np.array([-1,0,1])

for h in range(0,height):
    for w in range(1,width-2,1):
        temp = np.array([image[h][w],image[h][w+1],image[h][w+2]])
        ans = abs(np.dot(temp,conv).sum())
        
        if ans>cd:
            image[h][w] = 200 #ans갑을 그대로 넣어도 되지만 좀더 가시화 하기위해 200
        else:
            image[h][w]=0
            
            
cv2.imshow("moon",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
