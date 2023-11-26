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
frame = cv2.imread(f"img/{2}.jpg")
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
        cv2.line(transformed_image, (0, int(step)*j+6), (600, int(step)*j+6), color=(0,255,0))
    imshow_(transformed_image)
# %%

def assign_positions_grid(intersections):
    grid = {}
    step = int(600/19)
 
    for i in range(0, 20):
        for j in range(0, 20):
            for intersection in intersections:
                if int(step)*i+6 < intersection[0] and int(step)*(i+1)+7 > intersection[0] and int(step)*j+7 < intersection[1] and int(step)*(j+1)+6 > intersection[1]:
                    grid[(i, j)] = intersection
                if int(step)*(1)+7 > intersection[0] and int(step)*j+7 < intersection[1] and int(step)*(j+1)+6 > intersection[1]:
                    grid[(0, j)] = intersection
                if int(step)*19+6 < intersection[0] and int(step)*j+7 < intersection[1] and int(step)*(j+1)+6 > intersection[1]:
                    grid[(19, j)] = intersection
                if int(step)*i+6 < intersection[0] and int(step)*(i+1)+7 > intersection[0] and int(step)*(1)+6 > intersection[1]:
                    grid[(i, 0)] = intersection
                if int(step)*i+6 < intersection[0] and int(step)*(i+1)+7 > intersection[0] and int(step)*19+7 < intersection[1]:
                    grid[(i, 19)] = intersection
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
from sklearn.cluster import kMeans
# def assign_stones_to_positions(intersections, stones):
#     grid = assign_positions_grid(intersections)
#     map = {}
#     all_positions = np.concatenate
#     for key in grid:

# cls

#%%
grid = assign_positions_grid(all_intersections)
map = {}
black_stones = get_key_points(model_results, 0, perspective_matrix)
white_stones = get_key_points(model_results, 6, perspective_matrix)
vertical_lines, horizontal_lines = lines_detection(model_results, perspective_matrix)

vertical_lines = removeDuplicates(vertical_lines)
vertical_lines = restore_missing_lines(vertical_lines)

horizontal_lines = removeDuplicates(horizontal_lines)
horizontal_lines = restore_missing_lines(horizontal_lines)
        
cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]

intersections = detect_intersections(cluster_1, cluster_2, transformed_image)
#%%
all_positions = np.concatenate((black_stones, intersections))
# %%
clusters = KMeans(361).fit(all_positions)
# %%
labels = clusters.labels_
centroids = clusters.cluster_centers_

# %%
board = {}
split_index = len(black_stones)

for i in range(0, split_index):
    for j in range(split_index, len(all_positions)):
        if labels[i] == labels[j]:
            board[tuple(all_positions[j])] = all_positions[i]
        else:
            board[tuple(all_positions[j])] = None
grid = 
# %%
img = transformed_image.copy()
for key in board:
    if board[key] is not None:
        daw_points(np.array([board[key]]), img)
        cv2.putText(img, f'{key}', np.array(board[key]).astype(int), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
imshow_(img)
# %%
def is_stone_captured(sgf_path, move_number, coordinate):
        with open(sgf_path, 'rb') as f:
            collection = sgf.parse(f.read().decode('utf-8'))

        game_tree = collection.children[0]  # Assuming there is only one game tree in the SGF

        # Iterate through the moves until the specified move number
        for index, node in enumerate(game_tree.main_sequence):
            if index + 1 == move_number:
                # Check if the specified coordinate is occupied by a stone at the given move
                if coordinate in node.properties.get('B', []) or coordinate in node.properties.get('W', []):
                    # Check if the stone has liberties
                    liberties = get_liberties(node, coordinate)
                    return len(liberties) == 0

        return False

def get_liberties(node, coordinate):
    # Extract the board state from the SGF node
    board_size = int(node.properties.get('SZ', ['19'])[0])  # Assuming a default size of 19x19
    board = [[' ' for _ in range(board_size)] for _ in range(board_size)]

    # Fill in the stones from the SGF node
    for color, positions in [('B', node.properties.get('B', [])), ('W', node.properties.get('W', []))]:
        for pos in positions:
            row, col = sgf_coordinates_to_indices(pos)
            board[row][col] = color

    # Find the group to which the stone belongs
    group = find_group(board, coordinate)

    # Get liberties of the group
    liberties = set()
    for stone in group:
        liberties.update(get_adjacent_empty_positions(board, stone))

    return liberties

def sgf_coordinates_to_indices(sgf_coordinate):
    col = ord(su[0].upper()) - ord('A')
    row = int(sgf_coordinate[1:]) - 1
    return row, col

def find_group(board, start_position):
    color = board[start_position[0]][start_position[1]]
    group = set()
    visited = set()

    def dfs(position):
        if position in visited or board[position[0]][position[1]] != color:
            return
        visited.add(position)
        group.add(position)

        for neighbor in get_adjacent_positions(position, board_size=len(board)):
            dfs(neighbor)

    dfs(start_position)
    return group

def get_adjacent_positions(position, board_size):
    row, col = position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    adjacent_positions = [(row + dr, col + dc) for dr, dc in directions]

    return [(r, c) for r, c in adjacent_positions if 0 <= r < board_size and 0 <= c < board_size]

def get_adjacent_empty_positions(board, position):
    empty_positions = []
    for neighbor in get_adjacent_positions(position, board_size=len(board)):
        if board[neighbor[0]][neighbor[1]] == ' ':
            empty_positions.append(neighbor)
    return empty_positions

# Example usage
sgf_file_path = 'path/to/your/game.sgf'
move_number_to_check = 42
stone_coordinate_to_check = 'dd'

result = is_stone_captured(sgf_file_path, move_number_to_check, stone_coordinate_to_check)
print(f"Is stone captured? {result}")