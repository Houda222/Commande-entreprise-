#%%
from mySgf import GoBoard, GoSgf
import numpy as np
from processing import *
from ultralytics import YOLO
import copy
#%%
class GoGame:
    
    def __init__(self, model):

        self.model = model
        self.results = None
        self.moves = []
        self.not_moves = []
        self.old_game = np.zeros((19, 19))
        self.game = np.zeros((19, 19))
        self.map = {}
        
        self.transformed_image = None
        self.annotated_frame = None
    
    def detect_new_move(self, old_moves, new_moves):
        """
        Find the new played move

        Parameters:
        -----------
        old_moves : list
        
        new_moves : list

        Returns:
        --------
        tuple
            the new move 
        """

        new_move = [move for move in new_moves if move not in old_moves]
        return new_move

    def update_moves(self, old_moves, new_moves):
        """
        Add the new move to moves in a way to keep the order of the played moves

        Parameters:
        -----------
        old_moves : list
        
        new_moves : list

        Returns:
        --------
        list
            new moves in the correct order 
        """

        if old_moves is not None:
            new_move = self.detect_new_move(old_moves, new_moves)
            return old_moves.append(new_move)
        else:
            return new_moves


    def process_frame(self, frame):
        
        self.results = self.model(frame)
        self.annotated_frame = self.results[0].plot(labels=False, conf=False)
        # imshow_(self.annotated_frame)
        
        input_points = get_corners(self.results)

        output_edge = 600
        output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

        perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
        self.transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
        
        vertical_lines, horizontal_lines = lines_detection(self.results, perspective_matrix)
        
        vertical_lines = removeDuplicates(vertical_lines)
        vertical_lines = restore_missing_lines(vertical_lines)
        
        horizontal_lines = removeDuplicates(horizontal_lines)
        horizontal_lines = restore_missing_lines(horizontal_lines)
        
        
        img = np.copy(self.transformed_image)
        # draw_lines(vertical_lines, img)
        img = np.copy(self.transformed_image)
        # draw_lines(horizontal_lines, img)
        
        black_stones = get_key_points(self.results, 0, perspective_matrix)
        white_stones = get_key_points(self.results, 6, perspective_matrix)

        cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
        cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
        
        intersections = detect_intersections(cluster_1, cluster_2, self.transformed_image)
        
        if len(intersections) == 0:
            raise Exception(">>>>>No intersection were found!")
        
        self.update_game(white_stones, black_stones, intersections)
        self.define_ordered_moves()
        sgf_ = GoSgf('a', 'b', self.not_moves)
        # sgf_ = GoSgf('a', 'b', self.moves)
        _, sgf_n = sgf_.createSgf()

        board = GoBoard(sgf_n)
        return board.final_position()
        
        # self.old_moves = self.moves
        # self.moves = self.define_moves(white_stones, black_stones, intersections)

        # self.moves = self.update_moves(self.old_moves, self.moves)
    
    def update_game(self, white_stones_transf, black_stones_transf, transformed_intersections):
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
        
        self.map = create_board(transformed_intersections)
        self.old_game = copy.deepcopy(self.game)

        transformed_intersections = np.array(list(self.map.keys()))
        self.not_moves = []

        for stone in white_stones_transf:
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            self.game[self.map[nearest_corner][1], self.map[nearest_corner][0]] = 1
            self.not_moves.append(("W", (self.map[nearest_corner][0], 18 - self.map[nearest_corner][1])))
            
                
        for stone in black_stones_transf:
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            self.game[self.map[nearest_corner][1], self.map[nearest_corner][0]] = 1000
            self.not_moves.append(("B", (self.map[nearest_corner][0], 18 - self.map[nearest_corner][1])))
        
    
    def define_ordered_moves(self):
        difference = self.game - self.old_game
        pos_black = np.where(difference == 1000)
        pos_white = np.where(difference == 1)
        if len(pos_black[0]) > 1 or len(pos_white[0]) > 1 or len(pos_black[0])+len(pos_white[0]) > 1:
            print("MORE THAN ONE STONE WAS ADDED")
            return
        if len(pos_black[0]) != 0:
            self.moves.append(('B', (pos_black[0][0], pos_black[1][0])))
            return
        if len(pos_black[0]) != 0: 
            self.moves.append(('W', (pos_white[0][0], pos_white[1][0])))
            return
        print("no moves detected")        
        

        
#%%
model = YOLO('model.pt')
game = GoGame(model)
for i in range(1, 15):

    frame = cv2.imread(f"img/{i}.jpg")
    # imshow_(game.process_frame(frame))
    # annotated_frame = game.results[0].plot(labels=False, conf=False)
    # imshow_(annotated_frame)
    # print(game.game)
    # print(game.moves)


# %%

# def restore_missing_lines(lines, distance_threshold=10):
#     # ax=0 : x axis / ax=1 : y axis
#     lines = np.sort(lines, axis=0)
#     distances = calculate_distances(lines)
#     if len(distances) <= 1:
#         return lines
#     mean_distance, distances = find_common_distance(distances)
    
#     restored_lines = []
    
#     for i in range(len(lines) - 1):
#         print(i, lines)
#         spacing = (np.linalg.norm(lines[i + 1][:2]-lines[i][:2]) + np.linalg.norm(lines[i + 1][2:]-lines[i][2:]))/2
        
#         if is_approx_multiple(spacing, mean_distance, distance_threshold):
#             print("aprrox_multi")
#             if spacing >= mean_distance:
#                 num_missing_lines = round(spacing / mean_distance) - 1
#                 print("big spacing", num_missing_lines)
#                 for j in range(1, num_missing_lines + 1):
#                     if is_vertical(*lines[i]):
#                         x1 = lines[i][0] + j * mean_distance
#                         y1 = lines[i][1]
#                         x2 = lines[i][2] + j * mean_distance
#                         y2 = lines[i][3]
#                     else:
#                         x1 = lines[i][0]
#                         y1 = lines[i][1] + j * mean_distance
#                         x2 = lines[i][2]
#                         y2 = lines[i][3] + j * mean_distance
#                     restored_lines.append([x1, y1, x2, y2])
#         else:
#             print("deleting", spacing, mean_distance)
#             lines = np.delete(lines, i+1, axis=0)
#             i -= 1
  
  
#     if len(restored_lines) != 0:
#         lines = np.append(lines, np.array(restored_lines, dtype=int), axis=0)
#     lines = np.sort(lines, axis=0)
    
#     return lines
# # %%
# restore_missing_lines(horizontal_lines)