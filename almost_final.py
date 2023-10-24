# %%
from roboflow import Roboflow

import math, os
import cv2
import numpy as np

from numpy.linalg import norm


from sgfmill import sgf, sgf_moves
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


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray,5,100,100)

    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    gray = cv2.convertScaleAbs(dst)

    blur = cv2.GaussianBlur(gray,(5,5), 3)
    gray = cv2.subtract(gray,blur)

    canny = Canny_(gray)

    lines = HoughLinesP_(canny).squeeze()
    for i in range(len(lines)):
        lines[i] = interpolate(*lines[i])

    return removeDuplicates(lines)

def detect_intersections(lines):
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        if is_vertical(*line):
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)

    vertical_lines.sort(key=lambda x: x[0])
    horizontal_lines.sort(key=lambda x: x[1])

    corners = []
    for i in range(len(vertical_lines)):
        for j in range(len(horizontal_lines)):
            corners.append((vertical_lines[i][0], horizontal_lines[j][1]))
    
    return corners

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


def create_board(corners, board_size=(19,19)):
    cleaned_corners = []
    for corner in corners:
        if corner[0] > 5 and corner[0] < 595 and corner[1] < 595 and corner[1] > 5:
            cleaned_corners.append(corner)
    
    
    cleaned_corners.sort(key=lambda x: (x[1], x[0]))
    
    
    board = {}
    for j in range(board_size[1]):
        for i in range(board_size[0]):
            board[cleaned_corners.pop(0)] = (i, j)
    
    return board

def define_moves(white_stones, black_stones, intersections):
    Board = create_board(intersections)
    moves = []

    for stone in white_stones:
        for corner in intersections:
            if abs(corner[0] - stone[0]) <= 15 and abs(corner[1] - stone[1]) <= 15:
                moves.append(("W", (Board[corner][0], 18 - Board[corner][1])))
                break
            
    for stone in black_stones:
        for corner in intersections:
            if abs(corner[0] - stone[0]) <= 15 and abs(corner[1] - stone[1]) <= 15:
                moves.append(("B", (Board[corner][0], 18 - Board[corner][1])))
                break
    
    return moves

def transform_image(image_path):
    rf = Roboflow(api_key="Uhwq8zNnGlq5BnlPmd6F")
    project = rf.workspace().project("go-xoex6")
    model = project.version(2).model

    # infer on a local image
    res = model.predict(image_path, confidence=10, overlap=30).json()

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
    
    input_points = np.array(corners, dtype=np.float32)

    output_width, output_height = 600, 600
    output_points = np.array([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)

    image = cv2.imread(image_path)
    
    return cv2.warpPerspective(image, perspective_matrix, (output_width, output_height))




# %%
image_path = "20231004_130820.jpg"
imshow_(cv2.imread(image_path))
image = transform_image(image_path)

print("transformed the image")

lines = detect_lines(image)
intersections = detect_intersections(lines)

cv2.imwrite("testing.jpg", image)

print("detecting stones")
stones = Stones("testing.jpg")
white_stones, black_stones = stones.detect_stones()
print("stones detected")

moves = define_moves(white_stones, black_stones, intersections)


to_sgf = GoSgf("noir", "blanc", moves)
to_sgf.createSgf()
draw_board = GoBoard("noir_blanc.sgf")
draw_board.current_position()
# %%
