#%%
import cv2
import numpy as np
from RgbClass import RgbInv
"""
Detect stones on the board using the watershed algorithm
"""
class Stones:
    def __init__(self, img_url:str):

        # read input image to detect white stones
        img_white = cv2.imread(img_url)

        #read the inverted image to detect black stones
        img_black = RgbInv(img_url)
        img_black = img_black.detect()

        #initialize the parameters
        self.white_image = img_white
        self.black_image = img_black
    
    def detect_stones(self):
        
        #convert image to binary
        def binary(img):
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # threshold to binary
            thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]

            return gray, thresh

        #extract the areas that we are sure they are stones
        def areas(thresh):
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 6)
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
            return sure_fg

        def positions(sure_fg):
            positions = []
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg,  4, cv2.CV_32S)
            # loop over the number of unique connected component labels
            for i in range(0, numLabels):
                (cX, cY) = centroids[i]
                # append the position (center of the component) to the position list
                positions.append((cX,cY))
            return positions
    
        #get the parameters
        _ , white_thresh = binary(self.white_image)
        white_sure_fg = areas(white_thresh)

        #find the position
        #we remove the first element because it represents the background and not an actual stone
        white_stones_positions = positions(white_sure_fg)[1:]
        
        #get the parameters
        _ , black_thresh = binary(self.black_image)
        black_sure_fg = areas(black_thresh)

        #find the position
        black_stones_positions = positions(black_sure_fg)[1:]

        return white_stones_positions, black_stones_positions
    

stones = Stones("transformed_image.jpg")
white, black = stones.detect_stones()
# %%
