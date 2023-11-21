#%%

import threading
import copy
import traceback
from ultralytics import YOLO
import cv2
from processing import *
from GoGame import *

def draw_lines(lines, img, color=(0, 0, 255), thickness=1):
    global image
    if img is None:
        img = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def daw_points(points, img, color=(0, 0, 255), thickness=5):
    global image
    if img is None:
        img = np.zeros_like(image)
    for point in points.astype(int):
        cv2.circle(img, point, 1, 255, thickness)

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
def average_distance(lines):
    distances = [line_distance(lines[i + 1], lines[i]) for i in range(len(lines) - 1)]
    mean_distance = np.average(distances)
    return mean_distance
 #%%   
model = YOLO('model.pt')
game = GoGame(model)
frame = cv2.imread(f"img/{1}.jpg")
model_results = model(frame)

input_points = get_corners(model_results)

output_edge = 600
output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)

empty_intersections = model_results[0].boxes.xywh[model_results[0].boxes.cls == 3][:,[0, 1]]
empty_corner = model_results[0].boxes.xywh[model_results[0].boxes.cls == 4][:,[0, 1]]
empty_edge = model_results[0].boxes.xywh[model_results[0].boxes.cls == 5][:,[0, 1]]


if not empty_intersections is None:
    if len(empty_intersections) != 0:
        empty_intersections = np.array(empty_intersections[:, [0, 1]])
        empty_intersections = cv2.perspectiveTransform(empty_intersections.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

if not empty_corner is None:
    if len(empty_corner) != 0:
        empty_corner = np.array(empty_corner[:, [0, 1]])
        empty_corner = cv2.perspectiveTransform(empty_corner.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

if not empty_edge is None:
    if len(empty_edge) != 0:
        empty_edge = np.array(empty_edge[:, [0, 1]])
        empty_edge = cv2.perspectiveTransform(empty_edge.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))
transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
all_intersections = np.concatenate((empty_intersections, empty_corner, empty_edge), axis=0)

all_intersections = all_intersections[(all_intersections[:, 0:2] >= 0).all(axis=1) & (all_intersections[:, 0:2] <= 600).all(axis=1)]


# %%
# remove a given number of random elements
num_elements_to_remove = 20

# Generate random indices to remove
indices_to_remove = np.random.choice(all_intersections.shape[0], num_elements_to_remove, replace=False)

# Remove the selected indices
all_intersections_test = np.delete(all_intersections, indices_to_remove, axis=0)

# %%
import numpy as np
from scipy.interpolate import interp1d

# Sample data (replace this with your actual data)
intersection_data = all_intersections_test

# Separate x and y coordinates for interpolation
x_coords, y_coords = zip(*intersection_data)

# Polynomial interpolation function for x and y coordinates
poly_interp_x = interp1d(x_coords, y_coords, kind='quadratic', fill_value="extrapolate")
poly_interp_y = interp1d(y_coords, x_coords, kind='quadratic', fill_value="extrapolate")

# Determine grid bounds
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

# Generate positions for all points on the grid
all_positions = np.array([(x, y) for x in range(int(min_x), int(max_x) + 1, 32) for y in range(int(min_y), int(max_y) + 1, 32)])
# # Generate positions for missing intersections using polynomial interpolation
# missing_positions = [(x, int(poly_interp_x(x))) for x in range(int(min_x), int(max_x) + 1) if x not in x_coords]

# # Combine original and missing data
# interpolated_data = intersection_data + missing_positions

#%%
img2 = transformed_image.copy()

daw_points(all_intersections_test, img2)
imshow_(img2)
# %%
img = transformed_image.copy()
daw_points(all_positions, img)
imshow_(img)
# %%
