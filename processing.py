import math
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from mySgf import GoBoard, GoSgf


def interpolate(x1, y1, x2, y2, image_width=600, image_height=600):
    """
    Stretch a line to fit the whole image using the line equation y = slope * x + b

    Args:
    -----------
    x1: float
        The start point of the line in X direction
    y1: float
        The start point of the line in Y direction
    x2: float
        The end point of the line in X direction
    y2: float
        The end point of the line in Y direction
    image_width: int
        Width of the image (default is 600)
    image_height: int
        Height of the image (default is 600)
    
    Returns:
    --------
    numpy.ndarray
        new calculated endpoints 
    """

    slope, b = line_equation(x1, y1, x2, y2)
    if slope == float('Inf'):
        new_endpoints = np.array([x1, 0, x1, image_height], dtype=np.uint32)
    elif slope == 0:
        new_endpoints = np.array([0, y1, image_width, y1], dtype=np.uint32)
    else:
        left_bound = (0, np.round(b))
        right_bound = (image_width, np.round(slope * image_width + b))
        upper_bound = (np.round(-b/slope), 0)
        lower_bound = (np.round((image_height - b) / slope), image_height)
        possible_bounds = {left_bound, right_bound, upper_bound, lower_bound}

        new_endpoints = np.array([], dtype=np.uint32)
        for bound in possible_bounds:
            x, y = bound
            if x > image_width or x < 0 or y < 0 or y > image_height:
                continue
            new_endpoints = np.append(new_endpoints, (x, y))

    return new_endpoints

def line_equation(x1, y1, x2, y2):
    """
    Calculate the slope and the intercept b in the line equation y = slope * x + b

    Args:
    -----------
    x1: float
        The start point of the line in X direction
    y1: float
        The start point of the line in Y direction
    x2: float
        The end point of the line in X direction
    y2: float
        The end point of the line in Y direction
    
    Returns:
    --------
    float
        slope of the line
    float 
        intercept of the line 
    """
    if x1 == x2:
        "if slope is infinite , y = x1 = c"
        slope = float('Inf')
        b = x1
    else:
        slope = (y2-y1) / (x2-x1)
        b = y1 - slope * x1
    return slope, b

def adress_lines(lines):
    """
    Sort the order of endpoints

    Args:
    -----------
    lines: list
        list of lines unsorted
    
    Returns:
    --------
    list
        sorted lines    
    """
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]
        if (x1 + y1) > (x2 + y2):
            x1, x2, y1, y2 = x2, x1, y2, y1
            lines[i] = x1, y1, x2, y2
    return lines

def are_similar(line1, line2, threshold=10):
    """
    Compare two lines and decide if they're almost the same line based on a certain threshold

    Args:
    -----------
    line1: numpy.ndarray
        4 elements array representing the first line [x1, y1, x2, y2]
    line2: numpy.ndarray
        4 elements array representing the second line [x1, y1, x2, y2]
    threshold: int
        Smallest the difference between 2 lines (default is 10)
    
    Returns:
    --------
    bool
        true if similar, else false    
    """
    return np.all(np.abs(line1 - line2) <= threshold)


def removeDuplicates(lines):
    """
    Group similar lines and take the average of each group to keep one line per group

    Args:
    -----------
    lines: list
        list of lines to be filtered
    
    Returns:
    --------
    numpy.ndarray
        filtered list of lines   
    """
    grouped_lines = {}
    for line in lines:
        x1, y1, x2, y2 = line
        found = False
        for key in grouped_lines.keys():
            for element in grouped_lines[key]:
                if are_similar(element, line, threshold=15):
                    grouped_lines[key] = grouped_lines[key] + [line]
                    found = True
                    break
        if not found:
            grouped_lines[(x1, y1, x2, y2)] = [line]

    final_lines2 = []
    second_dict = {}
    for key in grouped_lines.keys():
        mean_line = np.mean(grouped_lines[key], axis=0).astype(dtype=int)
        final_lines2.append(mean_line)
        second_dict[tuple(mean_line)] = [mean_line]
    
    for line in lines:
        x1, y1, x2, y2 = line
        found = False
        for key in second_dict.keys():
            if are_similar(key, line, threshold=10):
                second_dict[key] = second_dict[key] + [line]
                found = True
                break
            
    final_lines = []
    for key in second_dict.keys():
        mean_line = np.mean(second_dict[key], axis=0).astype(dtype=int)
        final_lines.append(mean_line)
    
    return np.array(final_lines).astype(np.int32)


def is_vertical(x1, y1, x2, y2):
    """
    Decide if a line is vertical or not

    Args:
    -----------
    x1: float
        The start point of the line in X direction
    y1: float
        The start point of the line in Y direction
    x2: float
        The end point of the line in X direction
    y2: float
        The end point of the line in Y direction
    
    Returns:
    --------
    bool
        true if vertical, else false    
    """
    return abs(x1 - x2) < 50 and abs(y1 - y2) > 50

def intersect(line1, line2):
    """
    Find the intersection of 2 lines

    Args:
    -----------
    line1: list
        list of start point x, start point y, end point x, end point y
    line2: list
        list of start point x, start point y, end point x, end point y

    Returns:
    --------
    numpy.ndarray
        x and y coordinates of the intersection    
    """
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
    return np.array([int(np.round(x)), int(np.round(y))])

def get_angle(x1, y1, x2, y2):
    """
    Calculate the angle in radian of the segment in the trigonometric circle

    Args:
    -----------
    x1: float
        The start point of the line in X direction
    y1: float
        The start point of the line in Y direction
    x2: float
        The end point of the line in X direction
    y2: float
        The end point of the line in Y direction

    Returns:
    --------
    float
        Angle in radians
    """
    if x1 != x2:
        angle = np.arctan((y2 - y1)/(x2 - x1))
    else:
        angle = math.pi / 2
    return angle


def clean_lines(lines, image_width, image_height):
    """
    Clean lines by removing duplicates and stretching short lines

    Args:
    -----------
    lines: numpy.ndarray
        List of lines to be cleaned
    image_width: int
        width of the image
    image_height: int

    Returns:
    --------
    numpy.ndarray
        filtered list of lines   
    """   
    
    for i in range(len(lines)):
        lines[i] = interpolate(*lines[i], image_width, image_height)
        
    lines = adress_lines(lines)
    return removeDuplicates(lines)

def get_angles(lines):
    """
    Calculate the angle in radian for each line

    Args:
    -----------
    lines: list
        List of lines of which we calculate the angles

    Returns:
    --------
    numpy.ndarray
        List of angles in radians
    """

    lines_angles = np.zeros((lines.shape[0], 1))
    for i in range(len(lines)):
        lines_angles[i] = get_angle(*lines[i])
    return lines_angles

def cluster_orientation(lines):
    """
    Classify lines into horizontal and vertical lines using Kmeans clustering algorithm

    Args:
    -----------
    lines: list
        list of lines to be classified

    Returns:
    --------
    list
        horizontal or vertical lines
    list
        horizontal if first list is vertical, else vertical
        
    """

    lines_angles = get_angles(lines)
    
    stretched_angles = lines_angles * 2
    angle_space = np.column_stack((np.cos(stretched_angles), np.sin(stretched_angles)))

    num_clusters = 2

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(angle_space)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_

    # Separate lines into two clusters based on cluster labels
    cluster_1 = lines[cluster_labels == 0]
    cluster_2 = lines[cluster_labels == 1]
    return cluster_1, cluster_2

def create_board(intersections, board_size=(19,19)):
    """
    Set up the board with 19x19=361 intersections 

    Args:
    -----------
    intersections: numpy.ndarray
        List of found and interpolated intersections
    board_size:
        Size of the board (default is 19x19)

    Returns:
    --------
    dict
        The board, in which each key correponds to an intersection and its value represents its coordinate on the board
        
    """

    cleaned_intersections = intersections.tolist()
    cleaned_intersections.sort(key=lambda x: (x[1], x[0]))
    
    board = {}
    for j in range(0, 19):
        row = cleaned_intersections[:19]
        cleaned_intersections = cleaned_intersections[19:]
        row.sort(key=lambda x: x[0])
        for i in range(19):
            if len(row) != 0:
                board[tuple(row.pop(0))] = (i, j)
    
    return board

def detect_intersections(cluster_1, cluster_2, image):
    """
    Detect intersections between vertical and horizontal line clusters.

    Args:
    -----------
    cluster_1 : numpy.ndarray
                Array of vertical lines represented by coordinates [x1, y1, x2, y2].
    cluster_2 : numpy.ndarray
                Array of horizontal lines represented by coordinates [x1, y1, x2, y2].
    image : numpy.ndarray
            Image array to define the boundary for intersection points.

    Returns:
    --------
    numpy.ndarray
        Array of intersection points between vertical and horizontal line clusters.
    """
    intersections = []
    
    for v_line in cluster_1:
        for h_line in cluster_2:
            inter = intersect(v_line, h_line)
            
            if all(image.shape[:1] > inter) and all(inter >= 0):
                intersections.append(tuple(inter.astype(dtype=int)))
    
    return np.array(intersections)


    
def define_moves(white_stones_transf, black_stones_transf, transformed_intersections):
    """
    Define game moves based on the positions of white and black stones.

    Args:
    -----------
    white_stones_transf : numpy.ndarray
                          Array of coordinates representing transformed positions of white stones.
    black_stones_transf : numpy.ndarray
                          Array of coordinates representing transformed positions of black stones.
    transformed_intersections : numpy.ndarray
                                Array of perspective transformed intersection coordinates.

    Returns:
    --------
    list : python list
        List of moves, where each move is a tuple containing a color ('W' for white or 'B' for black)
        and the corresponding board position.
    """
    
    Board = create_board(transformed_intersections)
    transformed_intersections = np.array(list(Board.keys()))
    moves = []

    for stone in white_stones_transf:
        nearest_corner = None
        closest_distance = 100000
        for inter in transformed_intersections:
            distance = math.dist(inter, stone)
            if distance < closest_distance:
                nearest_corner = tuple(inter)
                closest_distance = distance
        moves.append(("W", (Board[nearest_corner][0], 18 - Board[nearest_corner][1])))
          
            
    for stone in black_stones_transf:
        nearest_corner = None
        closest_distance = 100000
        for inter in transformed_intersections:
            distance = math.dist(inter, stone)
            if distance < closest_distance:
                nearest_corner = tuple(inter)
                closest_distance = distance
        moves.append(("B", (Board[nearest_corner][0], 18 - Board[nearest_corner][1])))
        
    return moves

def calculate_distances(lines):
    """
    Calculate distances between consecutive lines.

    Args:
    -----------
    lines : numpy.ndarray
            Array of lines represented by coordinates [x1, y1, x2, y2].

    Returns:
    --------
    list : numpy.ndarray
            List of distances between consecutive lines.
    """
    distances = [(np.linalg.norm(lines[i + 1][:2]-lines[i][:2]) + np.linalg.norm(lines[i + 1][2:]-lines[i][2:])) / 2 for i in range(len(lines) - 1)]
    return distances

def find_common_distance(distances, target_distance=30):
    """
    Find the common distance among a set of distances using DBSCAN clustering.

    Args:
    -----------
    distances : list
                List of distances to be clustered and analyzed.
    target_distance : float, optional
                      The target distance to find among the clusters (default=30).

    Returns:
    --------
    Tuple
        Tuple containing the mean of the distances in the cluster with the target distance
        and the distances in that cluster.
    """
    
    # Reshape distances into a column vector
    distances_ = np.array(distances).reshape((-1, 1))

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1, min_samples=1)
    labels = dbscan.fit_predict(distances_)
    
    means = np.array([])
    unique_labels = np.unique(labels)
    label_index = np.array([])
    
    # Calculate means for each cluster and store label and mean in arrays
    for label in unique_labels:
        means = np.append(means, np.mean(distances_[labels==label]))
        label_index = np.append(label_index, label)

    # Find the index of the cluster with the closest mean to the target distance
    index = np.argmin(np.abs(means - target_distance))
    
    # Return the mean of distances in the chosen cluster and the distances in that cluster
    return means[index], distances_[labels==label_index[index]]

def is_approx_multiple(value, base, threshold):
    """
    Check if a value is approximately a multiple of a given base within a specified threshold.

    Args:
    -----------
    value : float
            The value to check.
    base : float
           The base value for which the check is performed.
    threshold : float
                The maximum allowed deviation from being a multiple.

    Returns:
    --------
    bool
        True if the value is approximately a multiple of the base within the threshold, False otherwise.
    """
    return abs(value - round(value / base) * base) < threshold

def restore_missing_lines(lines, distance_threshold=10):
    """
    Restore missing lines in a set of lines based on a common distance.

    Args:
    -----------
    lines : numpy.ndarray of shape (-1, 4)
            Array of lines represented by coordinates [x1, y1, x2, y2].
    distance_threshold : int, optional
                        Maximum threshold for spacing deviation to consider missing lines (default=10).

    Returns:
    --------
    numpy.ndarray
        Array of lines with restored missing lines.
    """
    # ax=0 : x axis / ax=1 : y axis
    lines = np.sort(lines, axis=0)
    distances = calculate_distances(lines)
    
    # If there are less than or equal to 1 distance, no restoration needed
    if len(distances) <= 1:
        return lines
    
    # Find mean distance and distances array after removing outliers
    mean_distance, distances = find_common_distance(distances)
    restored_lines = []
    
    for i in range(len(lines) - 1):
        spacing = (np.linalg.norm(lines[i + 1][:2]-lines[i][:2]) + np.linalg.norm(lines[i + 1][2:]-lines[i][2:]))/2
        
        if is_approx_multiple(spacing, mean_distance, distance_threshold):
            num_missing_lines = round(spacing / mean_distance) - 1
            
            for j in range(1, num_missing_lines + 1):
                if is_vertical(*lines[i]):
                    x1 = lines[i][0] + j * mean_distance
                    y1 = lines[i][1]
                    x2 = lines[i][2] + j * mean_distance
                    y2 = lines[i][3]
                else:
                    x1 = lines[i][0]
                    y1 = lines[i][1] + j * mean_distance
                    x2 = lines[i][2]
                    y2 = lines[i][3] + j * mean_distance
                restored_lines.append([x1, y1, x2, y2])
    
    # Append the restored lines to the original lines array
    if len(restored_lines) != 0:
        lines = np.append(lines, np.array(restored_lines, dtype=int), axis=0)
    
    # Sort the lines array
    lines = np.sort(lines, axis=0)
    
    return lines


def non_max_suppression(boxes, overlapThresh=0.5):
    """
    Apply non-maximum suppression to eliminate redundant bounding boxes.

    Args:
    -----------
    boxes : numpy.ndarray
            Array of bounding boxes with coordinates [x1, y1, x2, y2].
    overlapThresh : float, optional
                    Threshold for overlap to consider bounding boxes as redundant (default=0.5).

    Returns:
    --------
    numpy.ndarray
        Array of picked bounding boxes after non-maximum suppression.
    """
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



def lines_detection(model_results, perspective_matrix):
    """
    Process model results to identify and cluster all intersections.

    Args:
    -----------
    model_results : numpy.ndarray
                    List of model results containing information about boxes.
    perspective_matrix : numpy.ndarray
                    Perspective transformation matrix
    
    Returns:
    --------
    Tuple of two numpy.ndarrays representing clustered vertical and horizontal lines.
    """

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

    all_intersections = np.concatenate((empty_intersections, empty_corner, empty_edge), axis=0)

    all_intersections = all_intersections[(all_intersections[:, 0:2] >= 0).all(axis=1) & (all_intersections[:, 0:2] <= 600).all(axis=1)]



    all_intersections = all_intersections[all_intersections[:, 0].argsort()]

    all_intersections_x = all_intersections[:,0].reshape((-1, 1))

    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_x)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    # Sort the labels based on their counts in decreasing order
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_unique_labels = unique_labels[sorted_indices]


    lines_equations = np.array([]).reshape((-1, 2))
    lines_points_length = np.array([])
    cluster_vertical = np.array([]).reshape((-1, 4))

    for i, label in enumerate(sorted_unique_labels):
        line = all_intersections[cluster_labels==label]
        # print(i, len(line), line)
        # draw_points(line.astype(int), img)
        if len(line) > 2:
            # line = line[np.argsort(line[:, 1])]
            slope, intercept = np.polyfit(line[:,1], line[:,0], 1) # on inverse x et y
            line_ = np.array([intercept, 0, slope * 600 + intercept, 600])# on iverse les x et y
            lines_equations = np.append(lines_equations, [[slope, intercept]], axis=0)
        else:
            if len(cluster_vertical) == 0:
                raise Exception(f">>>>>> Cannot reconstruct ALL VERTICAL LINES")
            elif len(line) < 1:
                raise Exception(f">>>>>> Cannot reconstruct vertical line at point {line}")
            else:
                x1, y1 = line[0]
                slope = np.average(lines_equations[:,0], weights=lines_points_length, axis=0)
                intercept = x1 - slope * y1
                line_ = np.array([intercept, 0, slope * 600 + intercept, 600])
                lines_equations = np.append(lines_equations, [[slope, intercept]], axis=0)
        lines_points_length = np.append(lines_points_length, [len(line)], axis=0)
        
        # x1, y1, x2, y2 = line_.astype(np.uint32)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # draw_points([(x1, y1), (x2, y2)], img)
        
        cluster_vertical = np.append(cluster_vertical, [line_], axis=0)
    cluster_vertical = adress_lines(cluster_vertical)
    cluster_vertical = np.sort(cluster_vertical, axis=0).astype(int)


    all_intersections = all_intersections[all_intersections[:, 1].argsort()]
    all_intersections_y = all_intersections[:,1].reshape((-1, 1))

    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_y)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    # Sort the labels based on their counts in decreasing order
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_unique_labels = unique_labels[sorted_indices]

    # img = np.copy(transformed_image)
    lines_equations = np.array([]).reshape((-1, 2))
    lines_points_length = np.array([])
    cluster_horizontal = np.array([]).reshape((-1, 4))

    for label in sorted_unique_labels:
        line = all_intersections[cluster_labels==label]
        
        if len(line) > 2:
            line = line[np.argsort(line[:, 0])]
            slope, intercept = np.polyfit(line[:,0], line[:,1], 1)
            line = np.array([0, intercept, 600, slope * 600 + intercept])
            lines_equations = np.append(lines_equations, [[slope, intercept]], axis=0)
        else:
            if len(cluster_horizontal) == 0:
                raise Exception(f">>>>>> Cannot reconstruct ALL HORIZONTAL LINES")
            elif len(line) < 1:
                raise Exception(f">>>>>> Cannot reconstruct line at point {line}")
            else:
                x1, y1 = line[0]
                slope = np.average(lines_equations[:,0], weights=lines_points_length, axis=0)
                intercept = y1 - slope * x1
                line = np.array([0, intercept, 600, slope * 600 + intercept])
                lines_equations = np.append(lines_equations, [[slope, intercept]], axis=0)
        lines_points_length = np.append(lines_points_length, [len(line)], axis=0)
        
        # x1, y1, x2, y2 = line.astype(int)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # draw_points([(x1, y1), (x2, y2)], img)
        cluster_horizontal = np.append(cluster_horizontal, [line], axis=0)
    cluster_horizontal = adress_lines(cluster_horizontal)
    cluster_horizontal = np.sort(cluster_horizontal, axis=0).astype(int)
 
    return np.array(cluster_vertical).reshape((-1, 4)), np.array(cluster_horizontal).reshape((-1, 4))
    
def get_corners(results):    
    corner_boxes = np.array(results[0].boxes.xyxy[results[0].boxes.cls == 2])

    corner_boxes = non_max_suppression(corner_boxes)

    if len(corner_boxes) != 4:
        raise Exception(f">>>>Incorrect number of corners! Detected {len(corner_boxes)} corners")

    corner_centers = ((corner_boxes[:,[0, 1]] + corner_boxes[:,[2, 3]])/2)
    
    corner_centers = corner_centers[corner_centers[:, 1].argsort()]
    
    upper = corner_centers[:2]
    lower = corner_centers[2:]
    
    upper = upper[upper[:, 0].argsort()]
    lower = lower[lower[:, 0].argsort()[::-1]]
    
    return np.concatenate((upper, lower)).astype(dtype=np.float32)

def get_key_points(results, class_, perspective_matrix, output_edge=600):
    key_points = results[0].boxes.xywh[results[0].boxes.cls == class_]

    if not key_points is None:
        if len(key_points) != 0:
            key_points = np.array(key_points[:, [0, 1]])
            key_points_transf = cv2.perspectiveTransform(key_points.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))
            return key_points_transf[(key_points_transf[:, 0:2] >= 0).all(axis=1) & (key_points_transf[:, 0:2] <= output_edge).all(axis=1)]

    return key_points



def process_frame(model, frame):
    results = model(frame)
    input_points = get_corners(results)

    output_edge = 600
    output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
    
    vertical_lines, horizontal_lines = lines_detection(results, perspective_matrix)
    
    black_stones = get_key_points(results, 0, perspective_matrix)
    white_stones = get_key_points(results, 6, perspective_matrix)

    cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
    cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
    
    intersections = detect_intersections(cluster_1, cluster_2, transformed_image)
    
    if len(intersections) == 0:
        raise Exception(">>>>>No intersection were found!")
        
    moves = define_moves(white_stones, black_stones, intersections)
    
    return moves


def show_board(model, frame):
    moves = process_frame(model, frame)
    sgf_ = GoSgf('a', 'b', moves)
    _, sgf_n = sgf_.createSgf()

    board = GoBoard(sgf_n)
    return board.final_position(), sgf_n

