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
            raise Exception("Fix please")

    def show(self,window_name="Image"):
        cv.imshow(window_name, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    ''' Will check whether a region of the image is homoegenous or not based on differece in extremes'''
    def isHomogenous(self, section, threshold):
        return (np.max(section) - np.min(section) <= threshold)
    
    
    def split_and_merge(self, section, threshold):

        def recursive_split(section):
            print(f"section shape {section.shape}")
            rows, cols = section.shape
            if rows <= 1 or cols <= 1:
                return np.zeros_like(section, dtype=np.uint8)

            if self.isHomogenous(section, threshold):
                return np.ones_like(section, dtype=np.uint8)
            
            center_x, center_y = cols // 2, rows // 2
            
            # Define regions/sections
            top_left = section[:center_y, :center_x]
            top_right = section[:center_y, center_x:]
            bottom_left = section[center_y:, :center_x]
            bottom_right = section[center_y:, center_x:]
            
            segmented_window = np.zeros_like(section, dtype=np.uint8)
            segmented_window[:center_y, :center_x] = recursive_split(top_left)
            segmented_window[:center_y, center_x:] = recursive_split(top_right)
            segmented_window[center_y:, :center_x] = recursive_split(bottom_left)
            segmented_window[center_y:, center_x:] = recursive_split(bottom_right)

            return segmented_window
        
        segmented_image = recursive_split(self.img)
        return segmented_image
         
S = Segmentation("./img/cars.png")
new_image = S.split_and_merge(S.img, 30)
cv.imshow("Segmented",new_image*255)
S.show()

