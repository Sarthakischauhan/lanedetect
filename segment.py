import cv2 as cv
import numpy as np

class Segmentation():
    def __init__(self, imgPath):
        self.img_path = imgPath
        self.width = 0
        self.height = 0
        # Load the image
        self.img = self.load_image(imgPath)
        if self.img.any():
            self.height, self.width = self.img.shape
        
    def load_image(self, path):
        try:
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            return image 
        except:
            print("Error loading the file, try checking the path")
            raise Exception("Fix please")

    def show(self, window_name="Image"):
        cv.imshow(window_name, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def isHomogenous(self, section, threshold):
        std = np.std(section)
        return std <= threshold 
    
    def split_and_merge(self, section, split_threshold, merge_threshold):

        def recursive_split(section):
            rows, cols = section.shape
            if rows <= 1 or cols <= 1:
                return np.zeros_like(section, dtype=np.uint8)

            if self.isHomogenous(section, split_threshold):
                return np.ones_like(section, dtype=np.uint8) * np.mean(section)
            
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

        # Perform initial segmentation through recursive split
        segmented_image = recursive_split(self.img)
        
        # Merge adjacent similar regions
        merged_image = self.merge_regions(segmented_image, merge_threshold)
        print(merged_image)
        return merged_image

    ''' Merge the region in the block of 2x2'''
    def merge_regions(self, segmented_image, merge_threshold):
        rows, cols = segmented_image.shape
        merged_image = segmented_image.copy()

        def should_merge(region1, region2):
            return abs(np.mean(region1) - np.mean(region2)) < merge_threshold

        for i in range(0, rows - 1, 2):  
            for j in range(0, cols - 1, 2):
                top_left = merged_image[i:i + 2, j:j + 2]
                if i + 2 < rows:
                    bottom = merged_image[i + 2:i + 4, j:j + 2]
                    if should_merge(top_left, bottom):
                        merged_image[i:i + 4, j:j + 2] = np.mean(top_left)

                if j + 2 < cols:
                    right = merged_image[i:i + 2, j + 2:j + 4]
                    if should_merge(top_left, right):
                        merged_image[i:i + 2, j:j + 4] = np.mean(top_left)

        return merged_image

    ''' Split image into small sections using processlist method'''
    def split_processList(self, section, threshold):
        
        processList = []

        def recursive_split(section,x, y, w, h):
            # Add the processed segment in the array 
            rows, cols = section.shape
            nonlocal processList
            if rows <= 1 or cols <= 1:
                return np.zeros_like(section, dtype=np.uint8)

            if self.isHomogenous(section, split_threshold):
                processList.append([x, y, w, h, np.std(section)])
                return np.ones_like(section, dtype=np.uint8)

            center_x, center_y = cols // 2, rows // 2 
            # Define regions/sections
            top_left = section[:center_y, :center_x]
            top_right = section[:center_y, center_x:]
            bottom_left = section[center_y:, :center_x]
            bottom_right = section[center_y:, center_x:]
            
            segmented_window = np.zeros_like(section, dtype=np.uint8)
            segmented_window[:center_y, :center_x] = recursive_split(top_left,x,y,center_x, center_y, )
            segmented_window[:center_y, center_x:] = recursive_split(top_right, x+center_x, y, cols - center_x, center_y)
            segmented_window[center_y:, :center_x] = recursive_split(bottom_left,x, y+center_y,center_x, rows - center_y)
            segmented_window[center_y:, center_x:] = recursive_split(bottom_right, x+center_x, y+center_y, cols - center_x, rows - center_y )
            
            return segmented_window
        recursive_split(self.img, 0,0, self.width, self.height)
        return processList 

# Initialize the segmentation with image path
path = "./img/cars.png"
S = Segmentation(path)
# Perform split and merge with given thresholds
split_threshold = 10
merge_threshold = 30
new_image = S.split_and_merge(S.img, split_threshold, merge_threshold)

mask = (new_image > 0).astype(np.uint8) * 255

color_image = cv.imread(path)
# Create an overlay for segmented areas (red color)
overlay = np.zeros_like(color_image)
overlay[mask == 0] = (0, 0, 255)  # Red color

# Blend the original image with the overlay
alpha = 0.5 
result = cv.addWeighted(color_image, 1 - alpha, overlay, 1, 0)

# Display the final overlay result
cv.imshow("Segmented Overlay", result)
cv.waitKey(0)
cv.destroyAllWindows()

