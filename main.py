import math, os
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ultralytics import YOLO
from mySgf import GoBoard, GoSgf
import threading
import copy
import traceback

def interpolate(x1, y1, x2, y2, image_width=600, image_height=600):
    """
    Stretch a line to fit the whole image using the line equation y = slope * x + b

    Parameters:
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
    numpy array
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

    Parameters:
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

    Parameters:
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
    return np.all(np.abs(line1 - line2) <= threshold)


def removeDuplicates(lines):
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
    return np.array([int(np.round(x)), int(np.round(y))])

def get_angle(x1, y1, x2, y2):
    # return angle in radian
    if x1 != x2:
        angle = np.arctan((y2 - y1)/(x2 - x1))
    else:
        angle = math.pi / 2
    return angle


def clean_lines(lines, image_width, image_height):
    global image
    
    for i in range(len(lines)):
        lines[i] = interpolate(*lines[i], image_width, image_height)
        
    lines = adress_lines(lines)
    return  removeDuplicates(lines)

def get_angles(lines):
    lines_angles = np.zeros((lines.shape[0], 1))
    for i in range(len(lines)):
        lines_angles[i] = get_angle(*lines[i])
    return lines_angles

def cluster_orientation(lines):
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
    intersections = []
    
    for v_line in cluster_1:
        for h_line in cluster_2:
            inter = intersect(v_line, h_line)
            
            if all(image.shape[:1] > inter) and all(inter >= 0):
                intersections.append(tuple(inter.astype(dtype=int)))
    
    return np.array(intersections)


    
def define_moves(white_stones_transf, black_stones_transf, transformed_intersections):
    global transformed_image
    
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
    distances = [(np.linalg.norm(lines[i + 1][:2]-lines[i][:2]) + np.linalg.norm(lines[i + 1][2:]-lines[i][2:])) / 2 for i in range(len(lines) - 1)]
    return distances

def find_common_distance(distances, target_distance=30):
    distances_ = np.array(distances).reshape((-1, 1))

    dbscan = DBSCAN(eps=1, min_samples=1)
    labels = dbscan.fit_predict(distances_)
    
    means = np.array([])
    unique_labels = np.unique(labels)
    label_index = np.array([])
    
    for label in unique_labels:
        means = np.append(means, np.mean(distances_[labels==label]))
        label_index = np.append(label_index, label)

    index = np.argmin(np.abs(means - target_distance))
    
    return means[index], distances_[labels==label_index[index]]

def is_approx_multiple(value, base, threshold):
    return abs(value - round(value / base) * base) < threshold

def restore_missing_lines(lines, distance_threshold=10):
    # ax=0 : x axis / ax=1 : y axis
    lines = np.sort(lines, axis=0)
    distances = calculate_distances(lines)
    if len(distances) <= 1:
        return lines
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
    

    if len(restored_lines) != 0:
        lines = np.append(lines, np.array(restored_lines, dtype=int), axis=0)
    lines = np.sort(lines, axis=0)
    
    return lines


def non_max_suppression(boxes, overlapThresh=0.5):
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



def model_processing(model_results, perspective_matrix, frame):
    global cluster_vertical, cluster_horizontal, all_intersections, empty_intersections, empty_corner, empty_edge
    from matplotlib import pyplot as plt
    
    empty_intersections = model_results[0].boxes.xywh[results[0].boxes.cls == 3][:,[0, 1]]
    empty_corner = model_results[0].boxes.xywh[results[0].boxes.cls == 4][:,[0, 1]]
    empty_edge = model_results[0].boxes.xywh[results[0].boxes.cls == 5][:,[0, 1]]


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
    all_intersections_x = all_intersections[:,0].reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_x)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_
    unique_labels = np.unique(cluster_labels)
    
    cluster_vertical = []
    i = 0
    for label in unique_labels:
        line = all_intersections[cluster_labels==label]
        line = line[np.argsort(line[:, 1])]
        first_endpoint = line[0]
        last_endpoint = line[-1]
        line = interpolate(*first_endpoint, *last_endpoint)
        cluster_vertical.append(line)
        
    all_intersections_y = all_intersections[:,1].reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_y)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_
    unique_labels = np.unique(cluster_labels)
    

    cluster_horizontal = []
    i = 0
    for label in unique_labels:
        line = all_intersections[cluster_labels==label]
        line = line[np.argsort(line[:, 0])]
        first_endpoint = line[0]
        last_endpoint = line[-1]
        line = interpolate(*first_endpoint, *last_endpoint)
        cluster_horizontal.append(line)
    
    return np.array(cluster_vertical).reshape((-1, 4)), np.array(cluster_horizontal).reshape((-1, 4))
    

def master(frame):
    results = model(frame)
    
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
    
    input_points = np.concatenate((upper, lower)).astype(dtype=np.float32)

    output_edge = 600
    output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)

    transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
    
    vertical_lines, horizontal_lines = model_processing(results, perspective_matrix, transformed_image)
    
    black_stones = results[0].boxes.xywh[results[0].boxes.cls == 0]
    white_stones = results[0].boxes.xywh[results[0].boxes.cls == 6]

    black_stones = np.array(black_stones[:, [0, 1]])
    white_stones = np.array(white_stones[:, [0, 1]])

    black_stones_transf = cv2.perspectiveTransform(black_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))
    white_stones_transf = cv2.perspectiveTransform(white_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

    black_stones_transf = black_stones_transf[(black_stones_transf[:, 0:2] >= 0).all(axis=1) & (black_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
    white_stones_transf = white_stones_transf[(white_stones_transf[:, 0:2] >= 0).all(axis=1) & (white_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
    
       
    cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
    cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
    
    
    intersections = detect_intersections(cluster_1, cluster_2, transformed_image)
    
    if len(intersections) == 0:
        raise Exception(">>>>>No intersection were found!")

    moves = define_moves(white_stones_transf, black_stones_transf, intersections)

    sgf_ = GoSgf('a', 'b', moves)
    _, sgf_n = sgf_.createSgf()
    
    board = GoBoard(sgf_n)
    return board.final_position(), sgf_n



def process_frames():
    global ProcessFrame
    while True:
        if not ProcessFrame is None:
            try:
                
                ############ WA SMA3NI MZZZZN DB.  game_plot HYA LA VARIABLE LLI FIHA L'IMAGE DESSINé
                ############ B LE CODE DYAL HOUDA;
                ############ O sgf_filename HOWA LE NOM DYAL LE FICHER SGF LLI T ENREGISTRA 
                ############ QUI CORRESPOND A game_plot
                game_plot, sgf_filename = master(ProcessFrame)
                cv2.imshow("master", game_plot)
                
            except OverflowError as e:
                print(f"Overflow Error: {e}")
                
            except Exception as e:
                print('empty frame', type(e), e.args, e)
                traceback.print_exc()
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop if 'q' is pressed

model = YOLO('best8B.pt')

ProcessFrame = None

process_thread = threading.Thread(target=process_frames, args=())
process_thread.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    ########################## frame HYA LA VARIABLE LLI FIHA CHAQUE IMAGE DYAL STREAM
    ########################## YA3NI LE FLUX DE VIDEO QUI DOIT ETRE STREAMé
    ProcessFrame = copy.deepcopy(frame)
    
    cv2.imshow('Video Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break 

cap.release()
cv2.destroyAllWindows()

