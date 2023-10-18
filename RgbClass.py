#%%
import cv2
import numpy as np
from numpy.linalg import norm

class Rgb:
    def __init__(self, image_url:str):
        
        #load the image
        self.image = cv2.imread(image_url)
        
        #convert to rgb
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        #iterate over pixels
    
    def detect(self):
        
        if self.image is not None:
            
            # Get the height and width of the image
            height, width, channels = self.image.shape

            #initialize the detected image
            detected_image = np.zeros_like(self.image)
            
            #define the color margin/error
            err_black = 130
            err_white = 160
            err_green = 80
            
            for y in range(height):
                for x in range(width):
                    # Get the BGR color values of the pixel
                    blue, green, red = self.image[y, x]
                    
                    #compare the colors of the original image
                    if norm(np.array([blue, green, red]) - np.array([0, 0, 0])) < err_black:
                        detected_blue, detected_green, detected_red = 0, 0, 0
                    elif norm(np.array([blue, green, red]) - np.array([255, 255, 255])) < err_white:
                        detected_blue, detected_green, detected_red  = 255, 255, 255
                    elif norm(np.array([blue, green, red]) - np.array([17, 242, 39])) < err_green:
                        detected_blue, detected_green, detected_red  = 17, 242, 39
                    else:
                        detected_blue, detected_green, detected_red  = 127, 173, 200 

                    #reconstruct the detected image with the newly assigned colors
                    detected_image[y, x] = detected_blue, detected_green, detected_red
    
        return detected_image


# rgb = Rgb("transformed_image.jpg")
# rgb = rgb.detect()
# cv2.imwrite("rgb.jpg", rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



class RgbInv:
    def __init__(self, image_url:str):
        
        #load the image
        self.image = cv2.imread(image_url)

        #convert to rgb
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        #iterate over pixels
    
    def detect(self):
        
        if self.image is not None:
            
            # Get the height and width of the image
            height, width, channels = self.image.shape

            #initialize the detected image
            detected_image = np.zeros_like(self.image)
            
            #define the color margin/error
            err_black = 160
            err_white = 130
            err_green = 80
            
            for y in range(height):
                for x in range(width):
                    # Get the BGR color values of the pixel
                    blue, green, red = self.image[y, x]
                    
                    #compare the colors of the original image
                    if norm(np.array([blue, green, red]) - np.array([0, 0, 0])) < err_black:
                        detected_blue, detected_green, detected_red = 255, 255, 255
                    elif norm(np.array([blue, green, red]) - np.array([255, 255, 255])) < err_white:
                        detected_blue, detected_green, detected_red  = 0, 0, 0
                    elif norm(np.array([blue, green, red]) - np.array([17, 242, 39])) < err_green:
                        detected_blue, detected_green, detected_red  = 17, 242, 39
                    else:
                        detected_blue, detected_green, detected_red  = 0, 0, 0

                    #reconstruct the detected image with the newly assigned colors
                    detected_image[y, x] = detected_blue, detected_green, detected_red
    
        return detected_image

# rgb = RgbInv("transformed_image2.jpg")
# rgb = rgb.detect()
# cv2.imwrite("rgbInv2.jpg", rgb)


# class Stones:
#     def __init__(self, rgb:Rgb, image_url:str):
        
#         self.rgb = rgb
#         self.image = cv2.imread(image_url)
        
#     def find_stones(self):
          
#         if self.rgb is not None:
            
#             # Get the height and width of the image
#             height, width, channels = self.rgb.shape
        
#             #initialize the positions list
#             min_radius = 5
#             max_radius = 20
#             stones = []
#             white_stones = []
#             black_stones = []

#             gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#             gray = cv2.bilateralFilter(gray,5,100,100)
#             dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
#             gray = cv2.convertScaleAbs(dst)
#             blur = cv2.GaussianBlur(gray,(5,5), 3)
#             gray = cv2.subtract(gray,blur)
#             canny = cv2.Canny(gray, 50, 150)
#             # Find contours in the grayscale image
#             contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Create an empty image for drawing the contours
#             contour_image = np.zeros_like(self.image)

#             # Draw the contours on the empty image
            
#             detected_contours = []
#             # Iterate through the contours and filter by radius
#             for contour in contours:
#                 # Find the center and radius of the minimum enclosing circle
#                 (x, y), radius = cv2.minEnclosingCircle(contour)

#                 # Check if the radius falls within your desired range
#                 if (min_radius <= radius and radius <= max_radius):
#                     stones.append((x, y))
#                     detected_contours.append(contour)
#             cv2.drawContours(contour_image, detected_contours, -1, (0, 255, 0), 2)
#             cv2.imshow("a", contour_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
           
#             for (x, y) in stones:
#                 x, y = int(x), int(y)

#                 if (self.rgb[y, x] == (255, 255, 255)).all():
#                     white_stones.append((x, y))
#                 elif (self.rgb[y, x] == (0, 0, 0)).all():
#                     black_stones.append((x, y))
            
#         return white_stones, black_stones          

# stones = Stones(rgb, "5.jpg")
# white, black = stones.find_stones()
# print(len(white),len(black))
# %%
