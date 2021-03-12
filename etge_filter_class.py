#making conv filter (filter, padding, stride(need default) )
#height:행 , width:열 chennel:color층
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('C:/Users/dnfld/Desktop/deeplearning_learning/image/sunrise.PNG',cv2.IMREAD_GRAYSCALE)
height, width = image.shape

class picture:
    
    def __init__(self,fileName,cdf):
        self.name = fileName
        fileAddress = 'C:/Users/dnfld/Desktop/deeplearning_learning/image/'
        self.filter = np.array([-1,0,1]) #filter ndarray
        self.image = cv2.imread(fileAddress+fileName+'.PNG',cv2.IMREAD_GRAYSCALE)
        self.height ,self.width = self.image.shape
        self.cdf = 10 #cognition default

    def filt(self):
        for h in range(0,self.height):
            for w in range(1,self.width-2,1):
                temp = np.array([self.image[h][w],self.image[h][w+1],self.image[h][w+2]])
                ans = abs(np.dot(temp,self.filter).sum())
                if ans>self.cdf:
                    self.image[h][w] = 200
                else:
                    self.image[h][w] = 0

pic = picture('moon',10)
print(pic.height, pic.width)
pic.filt()
print(pic.height, pic.width)
cv2.imshow(pic.name, pic.image)
cv2.waitKey(0)
cv2.destroyAllWindows()