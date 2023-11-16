#%%
import math, os
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ultralytics import YOLO
from mySgf import GoBoard, GoSgf

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

def findContours_(image, imageToDrawOn=None):
    # ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not imageToDrawOn is None:
        cv2.drawContours(imageToDrawOn, contours, -1, (0, 255, 0), 1)
    return contours, hierarchy

def HoughLinesP_(image, imageToDrawOn):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=20, maxLineGap=100)
    img = np.zeros_like(imageToDrawOn)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
    return lines, img

def Canny_(image):
    high_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    high_thresh = 150
    lowThresh = 50
    return cv2.Canny(image, lowThresh, high_thresh)

def cartesian_to_polar(x1, y1, x2, y2):
    # Calculate the angle theta in radians
    if x2 - x1 != 0:
        theta = math.atan((y2 - y1) / (x2 - x1))
    else:
        theta = math.pi / 2  # Vertical line, atan(inf) = pi/2
    
    # Calculate the radius r
    r = x1 * np.cos(theta) + y1 * np.sin(theta)
    
    return r, theta

def draw_lines(lines, img=None, color=(0, 0, 255), thickness=1):
    global image
    if img is None:
        img = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_points(points, img=None, color=(0, 0, 255), thickness=3):
    global image
    if img is None:
        img = np.zeros_like(image)
    for point in points:
        cv2.circle(img, point, 3, color, thickness)
    imshow_(img)

def interpolate(x1, y1, x2, y2, image_width=600, image_height=600):
    "y = slope * x + b"
    slope, b = line_equation(x1, y1, x2, y2)
    if slope == float('Inf'):
        final_bounds = np.array([x1, 0, x1, image_height], dtype=np.uint32)
    elif slope == 0:
        final_bounds = np.array([0, y1, image_width, y1], dtype=np.uint32)
    else:
        left_bound = (0, np.round(b))
        right_bound = (image_width, np.round(slope * image_width + b))
        upper_bound = (np.round(-b/slope), 0)
        lower_bound = (np.round((image_height - b) / slope), image_height)
        possible_bounds = {left_bound, right_bound, upper_bound, lower_bound}

        final_bounds = np.array([], dtype=np.uint32)
        for bound in possible_bounds:
            x, y = bound
            if x > image_width or x < 0 or y < 0 or y > image_height:
                continue
            final_bounds = np.append(final_bounds, (x, y))

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

def adress_lines(lines):
    # Sorts the order of endpoints
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]
        if (x1 + y1) > (x2 + y2):
            x1, x2, y1, y2 = x2, x1, y2, y1
            lines[i] = x1, y1, x2, y2
    return lines

def are_similar(line1, line2, threshold=10):
    return np.all(np.abs(line1 - line2) <= threshold)

# def removeDuplicates(lines):
#     grouped_lines = {}
#     for line in lines:
#         x1, y1, x2, y2 = line
#         found = False
#         for key in grouped_lines.keys():
#             if are_similar(key, line):
#                 grouped_lines[key] = grouped_lines[key] + [line]
#                 found = True
#                 break
#         if not found:
#             grouped_lines[(x1, y1, x2, y2)] = [line]

#     final_lines = []
#     for key in grouped_lines.keys():
#         final_lines.append(np.mean(grouped_lines[key], axis=0))
    
#     return np.array(final_lines).astype(np.int32)

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
                # print("key, element", key, line)
                second_dict[key] = second_dict[key] + [line]
                found = True
                break
        # if not found:
        #     grouped_lines[(x1, y1, x2, y2)] = [line]
    
    final_lines = []
    for key in second_dict.keys():
        mean_line = np.mean(second_dict[key], axis=0).astype(dtype=int)
        final_lines.append(mean_line)
    
    
    # print(np.sort(np.array(list(second_dict.keys())), axis=0))
    # print(np.sort(np.array(list(grouped_lines.keys())), axis=0))
    
    
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

#%%

def get_blurred_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray,5,100,100)
    
    # dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    # gray = cv2.convertScaleAbs(dst)

    # blur = cv2.GaussianBlur(gray,(5,5), 3)
    # gray = cv2.subtract(gray,blur)
    return image, gray

def clean_lines(lines, image_width, image_height):
    global image
    
    for i in range(len(lines)):
        lines[i] = interpolate(*lines[i], image_width, image_height)
        
    lines = adress_lines(lines)
    # print("clean", lines.shape)
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
    # for inter in intersections:
    #     if inter[0] > 2 and inter[0] < 598 and inter[1] < 598 and inter[1] > 2:
    #         cleaned_intersections.append(inter)
    
    
    cleaned_intersections.sort(key=lambda x: (x[1], x[0]))
    print("nbre of intersections", len(cleaned_intersections))
    
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
            print(adress_lines([v_line]), adress_lines([h_line]))
            inter = intersect(v_line, h_line)
            
            if all(image.shape[:1] > inter) and all(inter >= 0):
                intersections.append(tuple(inter.astype(dtype=int)))
    
    return np.array(intersections)

def draw_corners(intersections, img):
    intersections.sort(key= lambda x: x[0] + x[1])
    tl = intersections[0]
    br = intersections[-1]
    cv2.circle(img,tl,3,255,10)
    cv2.circle(img,br,3,255,10)

    intersections.sort(key= lambda x: image.shape[0] - x[0] + x[1])
    tr = intersections[0]
    bl = intersections[-1]
    cv2.circle(img,tr,3,255,10)
    cv2.circle(img,bl,3,255,10)
    
    corners = np.array([tl, tr, br, bl])

    cv2.polylines(img, [np.array([[tl, tr, br, bl]])], isClosed=True, color=(255,0,0), thickness=2)
    imshow_(img)
    return corners

def get_corner_from_box(intersections, box, location): # box = (x1, y1, x2, y2)
    candidates = []
    for inter in intersections:
        if box[0] <= inter[0] <= box[2] and box[1] <= inter[1] <= box[3]:
            candidates.append(inter)
    
    if len(candidates) == 0:
        raise Exception(f"no intersections ware found at {location} corner")
    if len(candidates) == 1:
        return candidates[0]
    
    if location == "NO":
        return min(candidates, key=lambda x: x[0] + x[1])
    if location == "SE":
        return max(candidates, key=lambda x: x[0] + x[1])
    if location == "NE":
        return min(candidates, key=lambda x: box[2] - x[0] + x[1])
    if location == "SO":
        return max(candidates, key=lambda x: box[2] - x[0] + x[1])
    
    raise Exception("Nothing was found")

def get_corners(intersections, boxes):
    box_no = min(boxes, key=lambda x: x[0] + x[1] + x[2] + x[3])
    box_se = max(boxes, key=lambda x: x[0] + x[1] + x[2] + x[3])
    box_ne = max(boxes, key=lambda x: x[0] - x[1] + x[2] - x[3])
    box_so = min(boxes, key=lambda x: x[0] - x[1] + x[2] - x[3])
    
    corner_no = get_corner_from_box(intersections, box_no, "NO")
    corner_se = get_corner_from_box(intersections, box_se, "SE")
    corner_ne = get_corner_from_box(intersections, box_ne, "NE")
    corner_so = get_corner_from_box(intersections, box_so, "SO")
    
    return np.array([corner_no, corner_ne, corner_se, corner_so])

def define_moves(white_stones_transf, black_stones_transf, transformed_intersections):
    global transformed_image
    
    Board = create_board(transformed_intersections)
    transformed_intersections = np.array(list(Board.keys()))
    moves = []
    
    # print(Board)

    for stone in white_stones_transf:
        
        # cv2.circle(transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
        
        nearest_corner = None
        closest_distance = 100000
        for inter in transformed_intersections:
            distance = math.dist(inter, stone)
            if distance < closest_distance:
                nearest_corner = tuple(inter)
                closest_distance = distance
        moves.append(("W", (Board[nearest_corner][0], 18 - Board[nearest_corner][1])))
        # cv2.line(transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
            
    for stone in black_stones_transf:
        
        # cv2.circle(transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
        
        nearest_corner = None
        closest_distance = 100000
        for inter in transformed_intersections:
            distance = math.dist(inter, stone)
            if distance < closest_distance:
                nearest_corner = tuple(inter)
                closest_distance = distance
        moves.append(("B", (Board[nearest_corner][0], 18 - Board[nearest_corner][1])))
        # cv2.line(transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
        
    return moves

# %%
##############################""
def calculate_distances(lines):
    distances = [(np.linalg.norm(lines[i + 1][:2]-lines[i][:2]) + np.linalg.norm(lines[i + 1][2:]-lines[i][2:])) / 2 for i in range(len(lines) - 1)]
    # distances = [np.linalg.norm(lines[i + 1][:2] - lines[i][:2]) for i in range(len(lines) - 1)]
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
        # plt.scatter(np.ones(len(distances_[labels==label]))*10, distances_[labels==label])

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

###################


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

#%%
# def get_four_corners(corner_centers):
#         eps = 30
#         result_corners = []

#         for i in range(len(corner_centers)):
#             valid_corner = True

#             for j in range(len(result_corners)):
#                 if np.linalg.norm(corner_centers[i] - result_corners[j]) < eps:
#                     valid_corner = False
#                     break

#             if valid_corner:
#                 result_corners.append(corner_centers[i])

#         return result_corners

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
    
    colors = [(240, 42, 189), (77, 110, 250), (151, 23, 16), (123, 87, 66), (205, 51, 191), (94, 215, 51), (10, 100, 203), (16, 223, 22), (130, 110, 245), (173, 85, 89), (212, 146, 129), (89, 42, 8), (208, 191, 72), (51, 197, 91), (111, 30, 187), (191, 192, 198), (187, 55, 123), (1, 224, 111), (67, 122, 173)]

    cluster_vertical = []
    i = 0
    for label in unique_labels:
        # print(all_intersections_x[cluster_labels==label])
        # plt.scatter(np.arange(len(all_intersections_x[cluster_labels==label])), all_intersections_x[cluster_labels==label])
        # draw_points(all_intersections[cluster_labels==label].astype(dtype=int), frame, colors[i])
        # i += 1
        line = all_intersections[cluster_labels==label]
        # print("1",line)
        line = line[np.argsort(line[:, 1])]
        # print("2",line)
        first_endpoint = line[0]
        last_endpoint = line[-1]
        line = interpolate(*first_endpoint, *last_endpoint)
        # print("3",line)
        cluster_vertical.append(line)
        # print(cluster_vertical)
        
    all_intersections_y = all_intersections[:,1].reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_y)

    # Get the cluster labels for each line
    cluster_labels = kmeans.labels_
    unique_labels = np.unique(cluster_labels)
    
    colors = [(240, 42, 189), (77, 110, 250), (151, 23, 16), (123, 87, 66), (205, 51, 191), (94, 215, 51), (10, 100, 203), (16, 223, 22), (130, 110, 245), (173, 85, 89), (212, 146, 129), (89, 42, 8), (208, 191, 72), (51, 197, 91), (111, 30, 187), (191, 192, 198), (187, 55, 123), (1, 224, 111), (67, 122, 173)]

    cluster_horizontal = []
    i = 0
    for label in unique_labels:
        # print(all_intersections_x[cluster_labels==label])
        # plt.scatter(np.arange(len(all_intersections_x[cluster_labels==label])), all_intersections_x[cluster_labels==label])
        # draw_points(all_intersections[cluster_labels==label].astype(dtype=int), frame, colors[i])
        # i += 1
        line = all_intersections[cluster_labels==label]
        line = line[np.argsort(line[:, 0])]
        first_endpoint = line[0]
        last_endpoint = line[-1]
        line = interpolate(*first_endpoint, *last_endpoint)
        cluster_horizontal.append(line)
    
    return np.array(cluster_vertical).reshape((-1, 4)), np.array(cluster_horizontal).reshape((-1, 4))
    

def master(frame):
    global model, results, white_stones_transf, transformed_image, perspective_matrix, all_intersections, cluster_2, cluster_1
    results = model(frame)
    
    cv2.imwrite("results/frame.jpg", frame)
    annotated_frame = results[0].plot(labels=False, conf=False)
    cv2.imwrite("results/annotated.jpg", annotated_frame)
    
    corner_boxes = np.array(results[0].boxes.xyxy[results[0].boxes.cls == 2])
    
    # print("into non max")
    corner_boxes = non_max_suppression(corner_boxes)
    
    # print(corner_boxes)
    if len(corner_boxes) != 4:
        raise Exception(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Incorrect number of corners! Detected {len(corner_boxes)} corners")

    
    corner_centers = ((corner_boxes[:,[0, 1]] + corner_boxes[:,[2, 3]])/2)


    
    # corners = corner_centers.tolist()
    corners = corner_centers
    
    # corners.sort(key=lambda x: x[1])
    # upper = corners[:2]
    # lower = corners[2:]
    # upper.sort(key=lambda x: x[0])
    # lower.sort(key=lambda x: x[0], reverse=True)
    # corners = upper + lower
    corners = corners[corners[:, 1].argsort()]
    
    upper = corners[:2]
    lower = corners[2:]
    
    upper = upper[upper[:, 0].argsort()]
    lower = lower[lower[:, 0].argsort()[::-1]]
    
    corners = np.concatenate((upper, lower))

    input_points = np.array(corners, dtype=np.float32)
    # print("input to perspective", input_points.shape)
    # input_points_copy = input_points.copy().astype(dtype=int)
    
    # frame_copy = frame.copy()
    # for point in input_points_copy:
    #     cv2.circle(frame_copy, point, 5, color=(0, 0, 255), thickness=5)
    #     imshow_(frame_copy)

    output_edge = 600 # square of 600 by 600
    output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)

    transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
    # imshow_(transformed_image)
    cv2.imwrite("results/transformed_image.jpg", transformed_image)
    
    vertical_lines, horizontal_lines = model_processing(results, perspective_matrix, transformed_image)
    
    black_stones = results[0].boxes.xywh[results[0].boxes.cls == 0]
    white_stones = results[0].boxes.xywh[results[0].boxes.cls == 6]

    black_stones = np.array(black_stones[:, [0, 1]])
    white_stones = np.array(white_stones[:, [0, 1]])

    black_stones_transf = cv2.perspectiveTransform(black_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))
    white_stones_transf = cv2.perspectiveTransform(white_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

    black_stones_transf = black_stones_transf[(black_stones_transf[:, 0:2] >= 0).all(axis=1) & (black_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
    white_stones_transf = white_stones_transf[(white_stones_transf[:, 0:2] >= 0).all(axis=1) & (white_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
    
        
    # image, gray = get_blurred_img(transformed_image)
    # print(1)

    # canny = Canny_(gray)
    # # return canny
    # print(2)

    # a = np.zeros_like(transformed_image)
    # lines_, a = HoughLinesP_(canny, transformed_image)
    # cv2.imwrite("results/houghline.jpg", a)
    # print(3)
    
    # if lines_ is None:
    #     raise Exception(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>no lines ware detected")
    
    # if len(lines_) == 0:
    #     raise Exception(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>no lines ware detected")

    # lines = clean_lines(lines_.reshape((-1, 4)), image_width=transformed_image.shape[1], image_height=transformed_image.shape[0])

    # print(4)

    # cluster_1, cluster_2 = cluster_orientation(lines)
    # print(5)
    cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
    cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
    
    draw_lines(cluster_1.astype(dtype=int), transformed_image, color=(0, 0, 255))
    draw_lines(cluster_2.astype(dtype=int), transformed_image, color=(0, 255, 0))
    
    cv2.imwrite("results/transformed_image.jpg", transformed_image)
    # #######
    # if len(cluster_1) > 1:
    #     cluster_1 = restore_missing_lines(cluster_1)

    # if len(cluster_2) > 1:
    #     cluster_2 = restore_missing_lines(cluster_2)
    # #######

    intersections = detect_intersections(cluster_1, cluster_2, transformed_image)
    
    if len(intersections) == 0:
        raise Exception(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>No intersection were found!")
    
    print(6)
    moves = define_moves(white_stones_transf, black_stones_transf, intersections)

    # # draw_points(np.append(white_stones_transf, black_stones_transf, axis=0).astype(dtype=int), image)
    
    sgf_ = GoSgf('a', 'b', moves)
    sgf_f, sgf_n = sgf_.createSgf()
    
    board = GoBoard(sgf_n)
    cv2.imwrite("results/final.jpg", board.final_position())
    
    # return transformed_image
    return board.final_position()

# %%
import cv2
import threading
import queue
from ultralytics import YOLO
import copy
import traceback

# %%
def process_frames(queue):
    global ProcessFrame
    while True:
        # if queue.not_empty:
        if not ProcessFrame is None:
            # frame = queue.get()  # Get a frame from the queue
            try:
                # frame = cv2.imread("frame.jpg")
                cv2.imshow("master", master(ProcessFrame))
                    # Load the saved image using OpenCV
                # image = cv2.imread("result_image.jpg")
                
                # results = model(frame)
                # cv2.imshow("Result Image", results[0].plot(labels=False, conf=False))

                # # Display the image using OpenCV
                # cv2.imshow("Result Image", image) # Process the frame
                
            except OverflowError as e:
                print(f"Overflow Error: {e}")
                
            except Exception as e:
                print('empty frame', type(e), e.args, e)
                traceback.print_exc()
            # cv2.imshow('Processed Stream', cv2.flip(frame, 1))  # Display the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop if 'q' is pressed

# %%

model = YOLO('best8B.pt')

frame_queue = queue.Queue()
ProcessFrame = None

process_thread = threading.Thread(target=process_frames, args=(frame_queue,))
process_thread.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0 for the default camera, change it if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame_queue.put(frame)
    # image, gray = get_blurred_img(frame)

    # canny = Canny_(gray)
    
    ProcessFrame = copy.deepcopy(frame)
    # try:
    #     results = model(frame)
    #     annotated_frame = results[0].plot(labels=False, conf=False)
    #     corner_boxes = np.array(results[0].boxes.xyxy[results[0].boxes.cls == 2])
    #     corner_boxes = non_max_suppression(corner_boxes)
    #     corner_centers = ((corner_boxes[:,[0, 1]] + corner_boxes[:,[2, 3]])/2)
    #     corners = corner_centers.tolist()
    #     corners.sort(key=lambda x: x[1])
    #     upper = corners[:2]
    #     lower = corners[2:]
    #     upper.sort(key=lambda x: x[0])
    #     lower.sort(key=lambda x: x[0], reverse=True)
    #     corners = upper + lower

    #     input_points = np.array(corners, dtype=np.float32)
    #     output_edge = 600
    #     output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)
    #     perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
        
    #     transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
    #     annotated_frame = cv2.warpPerspective(annotated_frame, perspective_matrix, (output_edge, output_edge))
       
        
    #     # black_stones = results[0].boxes.xywh[results[0].boxes.cls == 0]
    #     # white_stones = results[0].boxes.xywh[results[0].boxes.cls == 6]

    #     # black_stones = np.array(black_stones[:, [0, 1]])
    #     # white_stones = np.array(white_stones[:, [0, 1]])

    #     # black_stones_transf = cv2.perspectiveTransform(black_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))
    #     # white_stones_transf = cv2.perspectiveTransform(white_stones.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

    #     # black_stones_transf = black_stones_transf[(black_stones_transf[:, 0:2] >= 0).all(axis=1) & (black_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
    #     # white_stones_transf = white_stones_transf[(white_stones_transf[:, 0:2] >= 0).all(axis=1) & (white_stones_transf[:, 0:2] <= output_edge).all(axis=1)]
 
    #     # # # cv2.imwrite("frame.jpg", frame)
    #     # image, gray = get_blurred_img(transformed_image)

    #     # canny = cv2.warpPerspective(canny, perspective_matrix, (output_edge, output_edge))
    #     # lines_, img = HoughLinesP_(canny, image)
    #     # clean_lines_img = np.zeros_like(image)
    #     # clusters_img = np.copy(image)
    #     # clusters_img2 = np.copy(image)
    
    
    
        
    #     # lines = clean_lines(lines_.squeeze(), image_width=frame.shape[1], image_height=frame.shape[0])
    #     # # print(lines.squeeze())
    #     # for line in lines:
    #     #     x1, y1, x2, y2 = line
    #     #     cv2.line(clean_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        
    #     # cluster_1, cluster_2 = cluster_orientation(lines)
        
    #     # for line in cluster_1:
    #     #     x1, y1, x2, y2 = line
    #     #     cv2.line(clusters_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    #     # for line in cluster_2:
    #     #     x1, y1, x2, y2 = line
    #     #     cv2.line(clusters_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
    #     # if len(cluster_1) > 1:
    #     #     cluster_1 = restore_missing_lines(cluster_1)

    #     # if len(cluster_2) > 1:
    #     #     cluster_2 = restore_missing_lines(cluster_2)
            
    #     # for line in cluster_1:
    #     #     x1, y1, x2, y2 = line
    #     #     cv2.line(clusters_img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    #     # for line in cluster_2:
    #     #     x1, y1, x2, y2 = line
    #     #     cv2.line(clusters_img2, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
    #     # print("lines processed")
        
    #     # intersections = detect_intersections(cluster_1, cluster_2, gray)
    #     # moves = define_moves(white_stones_transf, black_stones_transf, intersections)
    #     # sgf_ = GoSgf('a', 'b', moves)
    #     # sgf_f, sgf_n = sgf_.createSgf()
    #     # board = GoBoard(sgf_n)
        
    #     # cv2.imshow('Final rendering', board.final_position())
        
        

    #     # cv2.imshow('hough', img)
    #     # cv2.imshow('clean lines', clean_lines_img)
    #     # cv2.imshow('clustering', clusters_img)
    #     # cv2.imshow('clustering with missing lines', clusters_img)
    #     # cv2.imshow('Transformed image', transformed_image)
    #     cv2.imshow('annotated image', annotated_frame)
        
            
    # except Exception as e:
    #     traceback.print_exc()

    # cv2.imshow('canny', canny)
    cv2.imshow('Video Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break 

cap.release()
cv2.destroyAllWindows()

# # %%



# def record():
#     global frame, cap
    
    
#       # 0 for the default camera, change it if needed
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         print("Resolution", frame.shape)

#         cv2.imshow('Video Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):

#             break 

#     cap.release()
#     cv2.destroyAllWindows()
    

# %%

# model = YOLO('best8B.pt')

# cap = cv2.VideoCapture(0)

# frame = None


# process_thread = threading.Thread(target=record, args=())
# process_thread.start()



# while True:
#     # if queue.not_empty:
#     if not frame is None:
#         # frame = queue.get()  # Get a frame from the queue
#         try:
#             cv2.imshow("master", master(frame))
#                 # Load the saved image using OpenCV
#             # image = cv2.imread("result_image.jpg")

#             # # Display the image using OpenCV
#             # cv2.imshow("Result Image", image) # Process the frame
#         except:
#             print('empty frame')
#         # cv2.imshow('Processed Stream', cv2.flip(frame, 1))  # Display the processed frame
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break  # Break the loop if 'q' is pressed
    

# cv2.destroyAllWindows()
# %%
