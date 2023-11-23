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
frame = cv2.imread(f"img/{3}.jpg")
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
num_elements_to_remove = 1

# Generate random indices to remove
indices_to_remove = np.random.choice(all_intersections.shape[0], num_elements_to_remove, replace=False)

# Remove the selected indices
all_intersections_test = np.delete(all_intersections, indices_to_remove, axis=0)

# %%
import numpy as np
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator

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
missing_positions = np.array([(x, int(poly_interp_x(x))) for x in range(int(min_x), int(max_x) + 1, 32) if x not in x_coords])

# # Combine original and missing data
# interpolated_data = intersection_data + missing_positions

#%%
img2 = transformed_image.copy()

daw_points(missing_positions, img2)
imshow_(img2)
# %%
img = transformed_image.copy()
daw_points(all_positions, img)
imshow_(img)

#%%
import matplotlib.pyplot as plt
grid_points = all_intersections_test
x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])

xg, yg = np.meshgrid(x, y, indexing='ij')
def ff(x, y):
    return x**2 + y**2
data = ff(xg, yg)
interp = RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
           s=60, c='k', label='data')

xx = np.linspace(-4, 9, 31)
yy = np.linspace(-4, 9, 31)
X, Y = np.meshgrid(xx, yy, indexing='ij')
# interpolator
ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')

# # ground truth
# ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
#                   alpha=0.4, label='ground truth')
# plt.legend()
# plt.show()

# %%
grid_points = all_intersections_test
x, y = 

xg, yg = np.meshgrid(x, y, indexing='ij')

data = np.ones_like(x)
interp = RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(data.ravel(),
           s=60, c='k', label='data')

xx = np.linspace(-4, 9, 31)
yy = np.linspace(-4, 9, 31)
X, Y = np.meshgrid(xx, yy, indexing='ij')
# interpolator
ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')

# # ground truth
# ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
#                   alpha=0.4, label='ground truth')
# plt.legend()
# plt.show()
# %%

import matplotlib.pyplot as plt
from scipy.interpolate import griddata


known_x = all_intersections[:, 0]
known_y = all_intersections[:, 1]
known_z = np.ones_like(known_x)
# Create an interpolation functio

x_grid, y_grid = np.meshgrid(np.linspace(min(known_x), max(known_x), 100), np.linspace(min(known_y), max(known_y), 100))
z_grid = griddata((known_x, known_y), known_z, (x_grid, y_grid), method='linear')
plt.scatter(known_x, known_y, c=known_z, cmap='viridis', label='Known points')
# plt.contour(x_grid, y_grid, z_grid, levels=15, linewidths=0.5, colors='k')
plt.colorbar(label='Interpolated values')
plt.legend()
plt.show()
# %%
model = YOLO('model.pt')
game = GoGame(model)
for i in range(1, 15):

    frame = cv2.imread(f"img/{i}.jpg")
    results = model(frame)
    annotated_frame = results[0].plot(labels=False, conf=False)
    # imshow_(self.annotated_frame)

    input_points = get_corners(results)

    output_edge = 600
    output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
    step = int(600/19)
    for j in range(0, 20):
        cv2.line(transformed_image, (int(step)*j+6, 0), (int(step)*j+6, 600), color=(0,0,255))
        cv2.line(transformed_image, (0, int(step)*j+5), (600, int(step)*j+5), color=(0,255,0))
    imshow_(transformed_image)


# %%

def assign_positions_grid(intersections):
    grid = {}
    step = int(600/19)
 
    for i in range(0, 20):
        for j in range(0, 20):
            for intersection in intersections:
                if int(step)*i+6 < intersection[0] and int(step)*(i+1)+6 > intersection[0] and int(step)*j+6 < intersection[1] and int(step)*(j+1)+6 > intersection[1]:
                    grid[(i, j)] = intersection
    return grid
# %%
grid = assign_positions_grid(all_intersections)

# %%
img = transformed_image.copy()
for key in grid:
    daw_points(np.array([grid[key]]), img)
    cv2.putText(img, f'{key}', grid[key] )
    imshow_(img)


    
# %%
model = YOLO('model.pt')

for i in range(1, 10):

    frame = cv2.imread(f"img/{i}.jpg")
    model_results = model(frame)
    annotated_frame = model_results[0].plot(labels=False, conf=False)
    # imshow_(self.annotated_frame)

    input_points = get_corners(model_results)

    output_edge = 600
    output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
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

    grid = assign_positions_grid(all_intersections)


    img = transformed_image.copy()
    for key in grid:
        daw_points(np.array([grid[key]]), img)
        cv2.putText(img, f'{key}', np.array(grid[key]).astype(int), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
    imshow_(img)
# %%
