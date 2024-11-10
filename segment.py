#! base/bin/python 

import cv2 as cv 
import numpy as np

class Segmentation():
    def __init__(self, imgPath):
        self.img_path = imgPath
        # In case of loading image
        self.img = self.load_image(imgPath)
        
    def load_image(self, path):
        try:
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            return image 
        except:
            print("Error loading the file, try checking the path")
            exit()

    def show(self,window_name="Image"):
        cv.imshow(window_name, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    ''' Will check whether a region of the image is homoegenous or not based on differece in extremes'''
    def isHomogenous(self, section, threshold):
        return (np.max(section) - np.min(section) <= threshold)
    
    
    def split_and_merge(self, section, threshold):
        rows, cols = section.shape

        if (self.isHomogenous(section, threshold):
            return np.ones_like(section, dtype=np.uint8)
        
        center_x, center_y = cols // 2, rows // 2

        # Define regions

        print(f"Center points are {center_x}, {center_y}")


S = Segmentation("./img/sample.png")
S.split_and_merge(S.img, 10)
S.show()

