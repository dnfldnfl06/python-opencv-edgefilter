#making conv filter (filter, padding, stride(need default) )
#height:행 , width:열 chennel:color층
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
class picture:
    def __init__(self,fileName):
        self.name = fileName
        fileAddress = 'C:/Users/dnfld/Desktop/deeplearning_learning/image/'
        self.sobel_h = np.array([[-1,0,1],  #sobel horizontal
                               [-2,0,2],
                               [-1,0,1]]) 
        self.sobel_v = np.array([[-1,-2,-1],  #sobel vertical
                                  [0,0,0],
                                  [1,2,1]]) 
        
        self.image = cv2.imread(fileAddress+fileName+'.PNG',cv2.IMREAD_GRAYSCALE)
        self.height ,self.width = self.image.shape
        
    def filt(self,sobel):
        operator = self.sobel_v
        if sobel is 'h':
            operator = self.sobel_h
        for h in range(0,self.height-2,1):
            for w in range(0,self.width-2,1):
                temp = np.array([[self.image[h][w],self.image[h][w+1],self.image[h][w+2]],
                                 [self.image[h+1][w],self.image[h+1][w+1],self.image[h+1][w+2]],
                                 [self.image[h+2][w],self.image[h+2][w+1],self.image[h+2][w+2]]
                                ])
                
                ans = np.dot(temp,operator).sum()*0.2+40
                self.image[h][w] = ans
    def norm(self,pic):
        for h in range(0,self.height):
            for w in range(0,self.width):
                self.image[h][w] = math.sqrt(pic.image[h][w]**2+self.image[h][w]**2)
                
pic1 = picture('lena')
pic2 = picture('lena')
pic1.filt('v')
pic2.filt('h')
pic1.norm(pic2)
cv2.imshow(pic1.name, pic1.image)
cv2.waitKey(0)
cv2.destroyAllWindows()