# %%

import math, os
import cv2
import numpy as np

# %%

def imshow_(image):
    screen_width, screen_height = 1920, 1080  # Replace with your screen resolution or use a library to detect it dynamically

    # Calculate the scaling factors for width and height
    width_scale = screen_width / float(image.shape[1])
    height_scale = screen_height / float(image.shape[0])

    # Choose the smaller scaling factor to fit the image within the screen dimensions
    scale = min(width_scale, height_scale)

    # Resize the image with the calculated scale
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Get the dimensions of the resized image
    window_width, window_height = resized_image.shape[1], resized_image.shape[0]

    # Create a window with the determined size
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('img', window_width, window_height)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
def findContours_(image, imageToDrawOn=None):
    # ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not imageToDrawOn is None:
        cv2.drawContours(imageToDrawOn, contours, -1, (0, 255, 0), 1)
    return contours, hierarchy

def HoughLinesP_(image, imageToDrawOn=None):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)
    if not imageToDrawOn is None:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(imageToDrawOn, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
    return lines

def DoG_(image):
    low_sigma = cv2.GaussianBlur(image,(5,5),0)
    high_sigma = cv2.GaussianBlur(image,(7,7),0)
    
    dog = low_sigma - high_sigma
    return dog

def Canny_(image):
    high_thresh, thresh_im = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    high_thresh = 150
    lowThresh = 50
    return cv2.Canny(image, lowThresh, high_thresh)

#%%
    
image = cv2.imread('transformed_image.jpg')

imshow_(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# %%
imshow_(gray)
# # improve contrast 
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# gray = clahe.apply(gray)
# %%
# connected_lines = cv2.dilate(a, np.ones((5, 5), np.uint8), iterations=1)
# imshow_(connected_lines)
# %%
gray = cv2.bilateralFilter(gray,15,100,100)
#%%
gray = cv2.GaussianBlur(gray,(15,15), 3)

# %%
gray = cv2.bitwise_not(gray)
imshow_(gray)
# %%
blur = cv2.GaussianBlur(gray,(5,5), 3)

# blur = cv2.bilateralFilter(gray,5,100,100)

gray = cv2.subtract(gray,blur)
imshow_(gray)
#%%
dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)

 # converting back to uint8
gray = cv2.convertScaleAbs(dst)
imshow_(gray)
#
#%%

# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow_(gray)

# %%
# imshow_(gray2)
# gray = cv2.bilateralFilter(gray,5,100,100)
# gray = cv2.GaussianBlur(gray,(3,3),0)

fast = cv2.FastFeatureDetector_create()
gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
# find and draw the keypoints
kp = fast.detect(gray,None)
img2 = cv2.drawKeypoints(gray, kp, None, color=(255,0,0))
imshow_(img2)
# %%
gray2 = np.copy(gray)

#%%
blank = np.zeros_like(image)
# %%
corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 10000, 0.01, 30)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(blank,(int(x),int(y)),3,255,-1)
imshow_(blank)


#%%

canny = Canny_(gray)
# canny= cv2.bilateralFilter(canny,5,100,100)

imshow_(canny)

# %%
b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(b, 1000000, 0.001, 30)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(b,(int(x),int(y)),3,255,-1)
imshow_(b)

# to_contour = np.zeros_like(image)
# findContours_(canny, to_contour)
# imshow_(to_contour)
# %%
img = np.copy(image)
# can = cv2.bitwise_not(canny)
contours, hierarchy  = findContours_(gray, image)

# img = cv2.drawContours(image, contours, -1, (0,255,75), 2)

for contour in contours:
    print(contour.shape, contour[0].shape)
    if len(contour) > 4:
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = cv2.fitEllipse(contour)
        major_axis, minor_axis = axes
        try:
            cv2.ellipse(img, ellipse, (0,0,255), 2)
        except:
            pass
    # epsilon = 0.01*cv2.arcLength(contour,True)
    # approx = cv2.approxPolyDP(contour,epsilon,True)
    # cv2.drawContours(img, [approx], -1, (0,0,255), 2)
    # if len(approx) > 4:
    #     cv2.putText()
# %%
imshow_(img)

# # canny = Canny_(gray)
# # imshow_(canny)

# %%
# b = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
houghline = np.zeros_like(image)
lines = HoughLinesP_(canny, houghline)
# imshow_(cv2.cvtColor(cv2.imread("res.jpg"), cv2.COLOR_BGR2GRAY))
# lines = HoughLinesP_(cv2.cvtColor(cv2.imread("res.jpg"), cv2.COLOR_BGR2GRAY), houghline)
imshow_(houghline)
houghline = cv2.cvtColor(houghline, cv2.COLOR_BGR2GRAY)
absHoughline = np.float32(houghline>0) * 255
res = np.maximum(canny - absHoughline, np.zeros_like(canny))

imshow_(res)
cv2.imwrite("res.jpg", res)

# %%
import math

def cartesian_to_polar(x1, y1, x2, y2):
    # Calculate the angle theta in radians
    if x2 - x1 != 0:
        theta = math.atan((y2 - y1) / (x2 - x1))
    else:
        theta = math.pi / 2  # Vertical line, atan(inf) = pi/2
    
    # Calculate the radius r
    r = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return r, theta

from matplotlib import pyplot as plt
houghline = np.zeros_like(image)
lines = HoughLinesP_(canny, houghline).squeeze()
r_lst = []
theta_lst = []
for line in lines:
    r, theta = cartesian_to_polar(*line)
    r_lst.append(r)
    theta_lst.append(theta)

plt.plot(r_lst, theta_lst)
# %%

image = cv2.imread('transformed_image.jpg')

imshow_(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray,15,100,100)

dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
gray = cv2.convertScaleAbs(dst)
imshow_(gray)

blur = cv2.GaussianBlur(gray,(5,5), 3)
gray = cv2.subtract(gray,blur)
imshow_(gray)

canny = Canny_(gray)

houghline = np.zeros_like(image)
lines = HoughLinesP_(canny, houghline)

theta_values = np.linspace(-np.pi / 2, np.pi / 2, 360)
rho_values = np.arange(-np.sqrt(2) * 600, np.sqrt(2) * 600, 1)
hough_space = np.zeros((len(rho_values), len(theta_values)))

for line in lines:
    x1, y1, x2, y2 = line[0]
    theta = np.arctan2(y2 - y1, x2 - x1)
    rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    theta_idx = np.argmin(np.abs(theta_values - theta))
    rho_idx = np.argmin(np.abs(rho_values - rho))
    hough_space[rho_idx, theta_idx] += 1

plt.imshow(hough_space)


# %%

# # directory_path = 'C:\\Users\\ezzak\\Desktop\\Study\\FISE A2\\Commande Entreprise\\board recogntion\\img'

# # # Get a list of all files in the directory
# # files = os.listdir(directory_path)

# # # Filter image files (you can add more extensions if needed)
# # image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

# # # Loop through the image files and open them using OpenCV
# # for image_file in image_files:
# #     # Create the full path to the image file
# #     image_path = os.path.join(directory_path, image_file)
# #     print(image_path)
# # print("end")
# # image = cv2.imread(image_path)
# # imshow_(image)
    
############################################################################""
# %%
img_name = "20231004_130820.jpg"

from roboflow import Roboflow
rf = Roboflow(api_key="Uhwq8zNnGlq5BnlPmd6F")
project = rf.workspace().project("go-xoex6")
model = project.version(2).model
#%%
# infer on a local image
res = model.predict(img_name, confidence=10, overlap=30).json()
# %%
corners = []
for feature in res["predictions"]:
    if feature['class'] == 'corner':
        corners.append((feature['x'], feature['y']))

corners.sort(key=lambda x: x[1])
upper = corners[:2]
lower = corners[2:]
upper.sort(key=lambda x: x[0])
lower.sort(key=lambda x: x[0], reverse=True)
corners = upper + lower
# %%
input_points = np.array(corners, dtype=np.float32)

output_width, output_height = 600, 600
output_points = np.array([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]], dtype=np.float32)

perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
#%%
image = cv2.imread(img_name)
# %%
transformed_image = cv2.warpPerspective(image, perspective_matrix, (output_width, output_height))

# %%
cv2.imwrite("transformed_image3.jpg", transformed_image)
# %%


image = cv2.imread('transformed_image3.jpg')
image_width, image_height = image.shape[1], image.shape[0]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray,5,100,100)

dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
gray = cv2.convertScaleAbs(dst)

blur = cv2.GaussianBlur(gray,(5,5), 3)
gray = cv2.subtract(gray,blur)

canny = Canny_(gray)

a = np.zeros_like(image)
HoughLinesP_(canny, a)
# %%
cv2.imwrite("houghline.jpg", a)
# %%


import cv2


# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray,5,100,100)

    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    # gray = cv2.convertScaleAbs(dst)

    # blur = cv2.GaussianBlur(gray,(5,5), 3)
    # gray = cv2.subtract(gray,blur)

    canny = Canny_(gray)

    a = np.zeros_like(frame)
    HoughLinesP_(canny, frame)
    cv2.imshow('Camera Feed', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("out")
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()




# %%
# import cv2

# def find_camera_index():
#     for index in range(1,20):  # Try up to 10 indices (adjust if needed)
#         cap = cv2.VideoCapture(index)
#         if cap.isOpened():
#             print(f"Camera found at index {index}")
#             cap.release()
#             return index
#         else:
#             print(f"No camera at index {index}")
#     return None

# # Find the correct camera index
# print("search index")
# camera_index = find_camera_index()
# print("camera index is:", camera_index)
# print("done searching")
# if camera_index is not None:
#     print("start capture")
#     # Open the external camera
#     cap = cv2.VideoCapture(camera_index)
#     print("capture")
#     # Check if the camera opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         exit()

#     while True:
#         # Read a frame from the camera
#         ret, frame = cap.read()

#         # Check if the frame was successfully captured
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Display the frame in a window
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         gray = cv2.bilateralFilter(gray,5,100,100)

#         dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
#         gray = cv2.convertScaleAbs(dst)

#         blur = cv2.GaussianBlur(gray,(5,5), 3)
#         gray = cv2.subtract(gray,blur)

#         canny = Canny_(gray)

#         # a = np.zeros_like(frame)
#         HoughLinesP_(canny, frame)
#         cv2.imshow('Camera Feed', frame)

#         # Break the loop if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close all windows
#     cap.release()
#     cv2.destroyAllWindows()
# else:
#     print("No camera found. Please check your camera connection.")


# %%
from numpy.linalg import norm
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
    

# %%
def interpolate(x1, y1, x2, y2):
    "y = slope * x + b"
    slope, b = line_equation(x1, y1, x2, y2)
    if slope == float('Inf'):
        final_bounds = np.array([x1, 0, x1, 600], dtype=np.uint32)
    elif slope == 0:
        final_bounds = np.array([0, y1, 600, y1], dtype=np.uint32)
    else:
        left_bound = (0, b)
        right_bound = (600, int(slope * 600 + b))
        upper_bound = (-b/slope, 0)
        lower_bound = ((600 - b) / slope, 600)
        possible_bounds = [left_bound, right_bound, upper_bound, lower_bound]
        
        final_bounds = np.array([], dtype=np.uint32)
        for bound in possible_bounds:
            x, y = bound
            if x > 600 or x < 0 or y < 0 or y > 600:
                continue
            final_bounds = np.append(final_bounds, (int(x), int(y)))
    return final_bounds

def line_equation(x1, y1, x2, y2):
    if x1 == x2:
        "if slope is infinite , y = x1 = c"
        slope = float('Inf')
        b = x1
    else:
        slope = (y2-y1) / (x2-x1)
        b = y1 - slope * x1
    return slope, b

# def are_similar(slope1, b1, slope2, b2):
#     print("slope1", slope1, "slope2", slope2)
#     slope1 = float("Inf") if abs(slope1) > 40 else slope1
#     slope2 = float("Inf") if abs(slope2) > 40 else slope2
#     print("slope1", slope1, "slope2", slope2, end="\n\n")
#     if abs(slope1) == float('Inf') and abs(slope2) == float('Inf'):
#         return abs(b1 - b2) <= 30

#     return abs(slope1 - slope2) <= 100 and abs(b1 - b2) <= 20

def are_similar(line1, line2):
    return abs(line1[0] - line2[0]) <= 15 and abs(line1[1] - line2[1]) <= 15

def removeDuplicates(lines):
    grouped_lines = {}
    for line in lines:
        x1, y1, x2, y2 = line
        found = False
        for key in grouped_lines.keys():
            if are_similar(key, line):
                grouped_lines[key] = grouped_lines[key] + [line]
                found = True
                break
        if not found:
            grouped_lines[(x1, y1, x2, y2)] = [line]

    final_lines = []
    for key in grouped_lines.keys():
        final_lines.append(grouped_lines[key][0])
    
    return np.array(final_lines)

def is_vertical(x1, y1, x2, y2):
    return abs(x1 - x2) < 50 and abs(y1 - y2) > 50

def intersect(line1, line2):
    slope1, b1 = line_equation(*line1)
    slope2, b2 = line_equation(*line2)
    if slope1 == float('Inf'):
        x = b1
        y = slope2 * x + b2
    elif slope2 == float('Inf'):
        x = b2
        y = slope1 * x + b1
    else:
        x = (b2 - b1) / (slope1 - slope2)
        y = slope1 * x + b1
    return int(x), int(y)

#%%
def create_board(corners, board_size=(19,19)):
    cleaned_corners = []
    for corner in corners:
        if corner[0] > 5 and corner[0] < 595 and corner[1] < 595 and corner[1] > 5:
            cleaned_corners.append(corner)
    
    
    cleaned_corners.sort(key=lambda x: (x[1], x[0]))
    print(cleaned_corners)
    
    board = {}
    for j in range(board_size[1]):
        for i in range(board_size[0]):
            board[cleaned_corners.pop(0)] = (i, j)
    
    return board

# %%

# def removeDuplicates(lines):
#     grouped_lines = {}
#     Slopes = []
#     for line in lines:
#         slope, b = line_equation(*line)
#         Slopes.append(slope)
#         # print("slope", slope, "b", b)
#         keys = grouped_lines.keys()
#         found = False
#         for key in grouped_lines.keys():
#             # print("testing similarity", key, (slope, b))
#             if are_similar(*key, *(slope, b)):
#                 grouped_lines[key] = grouped_lines[key] + [line]
#                 found = True
#                 # print("similar", grouped_lines[key])
#                 break
#             # print("not similar")
#         if not found:
#             grouped_lines[(slope, b)] = [line]
#             # print("not found", grouped_lines[(slope, b)])
    
#     # print("\n done grouping")
#     final_lines = []
#     for key in grouped_lines.keys():
#         # print(grouped_lines[key][0])
#         final_lines.append(grouped_lines[key][0])
#     Slopes.sort()
#     print(Slopes)
#     return np.array(final_lines)
    
# %%
image = cv2.imread('transformed_image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray1 = cv2.bilateralFilter(gray,5,100,100)

image = cv2.imread('transformed_image3.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray2 = cv2.bilateralFilter(gray,5,100,100)

imshow_(gray1)
imshow_(gray2)

    
# %%

image = cv2.imread('transformed_image3.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray,5,100,100)

dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
gray = cv2.convertScaleAbs(dst)

blur = cv2.GaussianBlur(gray,(5,5), 3)
gray = cv2.subtract(gray,blur)

canny = Canny_(gray)

a = np.zeros_like(image)
lines = HoughLinesP_(canny, a)

# imshow_(a)
# %%
b = np.zeros_like(image)
lines = HoughLinesP_(canny, a).squeeze()

for i in range(len(lines)):
    lines[i] = interpolate(*lines[i])

clean_lines = removeDuplicates(lines)

for line in clean_lines:
    x1, y1, x2, y2 = line
    cv2.line(b, (x1, y1), (x2, y2), (0, 0, 255), 1)

# imshow_(b)
# %%

vertical_lines = []
horizontal_lines = []
for line in clean_lines:
    if is_vertical(*line):
        vertical_lines.append(line)
    else:
        horizontal_lines.append(line)

vertical_lines.sort(key=lambda x: x[0])
horizontal_lines.sort(key=lambda x: x[1])
# %%
b2 = b.copy()
corners = []
for v_line in vertical_lines:
    for h_line in horizontal_lines:
        inter = intersect(v_line, h_line)
        corners.append(inter)
        cv2.circle(b2,inter,3,255,-1)

imshow_(b2)

# %%


stones = Stones("transformed_image3.jpg")
white_stones, black_stones = stones.detect_stones()
# %%
image2 = image.copy()
Board = create_board(corners)
moves = []

for stone in white_stones:
    cv2.circle(image2, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
    nearest_corner = None
    closest_distance = 100000
    for corner in corners:
        distance = math.dist(corner, stone)
        if distance < closest_distance:
            nearest_corner = corner
            closest_distance = distance
    moves.append(("W", (Board[corner][0], 18 - Board[corner][1])))
    cv2.line(image2, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
        
for stone in black_stones:
    cv2.circle(image2, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
    nearest_corner = None
    closest_distance = 100000
    for corner in corners:
        distance = math.dist(corner, stone)
        if distance < closest_distance:
            nearest_corner = corner
            closest_distance = distance
        
    moves.append(("B", (Board[nearest_corner][0], 18 - Board[nearest_corner][1])))
    cv2.line(image2, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
imshow_(image2)
# %%


from sgfmill import sgf, sgf_moves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
class Goboard: 
creates a board given an sgf file provided by the GoSgf class
can navigate through the game using methods such as previous or next
"""

class GoBoard:
    def __init__(self, sgf_url:str):
        with open(sgf_url, 'rb') as f:
            sgf_content = f.read()

        # Load an sgf file/ the game
        self.sgf_game = sgf.Sgf_game.from_bytes(sgf_content)
        
        #get the game size
        self.board_size = self.sgf_game.get_size()

        # Extract the game moves
        self.moves = []
        for node in self.sgf_game.get_main_sequence():
            color, move = node.get_move()
            if color is not None and move is not None:
                row, col = move
                self.moves.append((row, col, color)) 

        #get the number of moves 
        self.total_number_of_moves = len(self.moves)

        #define the current number of moves initialized by the total number, and which we'll modify each time whne calling the previus or the next fucntion
        self.current_number_of_moves = self.total_number_of_moves
        
       
       
    # Draw the board up to a certain number of moves
    def drawBoard(self, number_of_moves_to_show : int):
        board = np.zeros((self.board_size, self.board_size))
        fig, ax = plt.subplots(figsize=(8, 8))

        #extract the moves we wanna show
        extracted_moves = self.moves[:number_of_moves_to_show]

        #set up the board's background
        background = patches.Rectangle((-1,-1), self.board_size + 1, self.board_size + 1, facecolor='#EEAD0E', fill = True, edgecolor='black')
        ax.add_patch(background)

        # Draw lines for the board grid
        
        # for i in range(board_size):
        #     ax.plot([i, i], [0, board_size - 1], color='k', linewidth = 0.7)
        #     ax.plot([0, board_size - 1], [i, i], color='k', linewidth = 0.7)
        
        for i in range(self.board_size):
            # Vertical lines and letters
            ax.add_patch(patches.Rectangle((i - 0.01, -0.01), 0.02, self.board_size + 0.02 -1, color='k'))
            plt.text(i, -0.8, chr(97 + i), fontsize=8, color='black')       
            # Horizontal lines and letters
            ax.add_patch(patches.Rectangle((-0.01, i - 0.01), self.board_size + 0.02 - 1, 0.02, color='k'))
            plt.text(-0.8, i, chr(97 + i), fontsize=8, color='black')       


        # Set axis limits to include the entire grid
        ax.set_xlim(-1, self.board_size)
        ax.set_ylim(-1, self.board_size)

        # Draw stones
        for move in extracted_moves:
            row, col, color = move
            if board[row, col] == 0:
                stone_color = 'black' if color == 'b' else 'white'
                board[row, col] = 1
                ax.add_patch(plt.Circle((col, self.board_size - row - 1), 0.4, facecolor=stone_color, fill = True, edgecolor = 'black'))
        
        #setting the contour of the last move to a different color
        last_move = extracted_moves[-1]           
        stone_color = 'black' if last_move[2] == 'b' else 'white'
        ax.add_patch(patches.Circle((last_move[1], self.board_size - last_move[0] - 1), 0.4, facecolor=stone_color, fill = True, edgecolor = 'red'))

        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()



    #display the initial position with the first move
    def initial_position(self):
        self.current_number_of_moves = 1
        self.drawBoard(1)

    #display the final position 
    def final_position(self):
        self.current_number_of_moves = self.total_number_of_moves
        self.drawBoard(self.total_number_of_moves)

    #display the current position
    def current_position(self):
        self.drawBoard(self.current_number_of_moves)

    #display whose turn to play
    def current_turn(self):
        turn = self.moves[self.current_number_of_moves - 1][2]
        if turn == 'b':
            return 'White' 
        elif turn == 'w' or self.current_number_of_moves == 0:
            return 'black'
        
    #access to previous move
    def previous(self):
        if self.current_number_of_moves > 1:
            self.current_number_of_moves -= 1
            self.drawBoard(self.current_number_of_moves)


    #access to next move
    def next(self):
        if self.current_number_of_moves < self.total_number_of_moves:
            self.current_number_of_moves += 1
            self.drawBoard(self.current_number_of_moves)

    

            
"""
class GoSgf: 
creates an sgf file given the list of moves (stones and their positions) extracted from the image recognition part
"""

class GoSgf:
    def __init__(self, black:str, white:str, moves:list, tournament=None):
        # define the game information
        self.black = black
        self.white = white
        self.board_size = (19,19)
        self.tournament = tournament

        self.game_info = {
            "GM" : "1", #game type, 1 for go
            "EV" : tournament,
            "PB" : black,
            "PW" : white,
            #"SZ" : f"{self.board_size[0]}",
            #"KM" : "6.5", #komi
            #"RU" : "Japanese" #rules used
        }

        #get the moves we collected from board recognition
        self.moves = moves

    #write the sgf file
    def createSgf(self):
        
        #convert a move to SGF format
        def add_to_sgf(move):
            player, position = move
            x, y = position 
            sgf_x = chr(ord('a') + x)
            sgf_y = chr(ord('a') + y)
            return f";{player}[{sgf_x}{sgf_y}]"

        #convert the sgf file 
        def assembleSgf():
            sgf_ = ''.join([add_to_sgf(move) for move in self.moves])
            return sgf_
        
        sgf_moves = assembleSgf()
        sgf_filename = f"{self.black}_{self.white}.sgf"
        
        with open(sgf_filename, "w") as sgf_file:

            # Write game information
            sgf_file.write("(; \n")
            for key, value in self.game_info.items():
                sgf_file.write(f"{key}[{value}]\n")
            sgf_file.write("\n")

            # Write stone positions
            sgf_file.write(sgf_moves)

            # End the SGF file
            sgf_file.write(")\n")

        return sgf_file, sgf_filename



#moves = [("B", (3, 3)), ("W", (4, 4)), ("B", (5, 5)), ("W", (6, 6))]  # Example moves


# %%

to_sgf = GoSgf("noir", "blanc", moves)
to_sgf.createSgf()
draw_board = GoBoard("noir_blanc.sgf")
draw_board.current_position()
# %%
