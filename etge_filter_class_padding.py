#making conv filter (filter, padding, stride(need default) )
#height:행 , width:열 chennel:color층
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('C:/Users/dnfld/Desktop/deeplearning_learning/image/lena.PNG',cv2.IMREAD_GRAYSCALE)
height, width = image.shape

class picture:
    
    def __init__(self,fileName,cdf,padding):
        self.name = fileName
        fileAddress = 'C:/Users/dnfld/Desktop/deeplearning_learning/image/'
        self.filter = np.array([-1,0,1]) #filter ndarray
        self.image = cv2.imread(fileAddress+fileName+'.PNG',cv2.IMREAD_GRAYSCALE)
        self.padding = padding
        self.height ,self.width = self.image.shape
#       self.height+=padding this class filter is for width
        self.width=self.width+padding
        self.cdf = 50 #cognition default

    def filt(self):
        
        for h in range(0,self.height):
            for w in range(self.padding,self.width-self.padding,1):
                temp = np.array([self.image[h][w-1],self.image[h][w],self.image[h][w+1]])
                ans = abs(np.dot(temp,self.filter).sum())
                if ans>self.cdf:
                    self.image[h][w] = 200
                else:
                    self.image[h][w] = 0
        

    def pad_filter(self):
        padded_image = np.array([[0]*self.width]*self.height)
        for h in range(0,self.height):
            for w in range(self.padding,self.width-self.padding):
                padded_image[h][w] = image[h][w-self.padding]
        self.image = padded_image.astype(np.int8)
        
        
pic = picture('lena',10,1)
print(pic.height, pic.width)
pic.pad_filter()
pic.filt()
cv2.imshow(pic.name, pic.image)
cv2.waitKey(0)
cv2.destroyAllWindows()