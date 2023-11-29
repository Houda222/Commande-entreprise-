#%%
from mySgfCopy import GoBoard, GoSgf
import numpy as np
from processing import *
from ultralytics import YOLO, utils
import copy
import sente
#%%
class GoGame:
    STATES = []
    
    def __init__(self, model, sgf):

        self.model = model
        self.sgf = sgf
        self.results = None
        self.frame = None
        self.moves = []
        self.all_moves = []
        self.old_state = np.zeros((19, 19))
        self.state = np.zeros((19, 19))
        self.map = {}
        self.transformed_image = None
        self.annotated_frame = None
        
        self.game = sente.Game()
    
    def initialize_game(self, frame):
        self.frame = frame
        self.process_frame(frame)
        self.moves = copy.deepcopy(self.all_moves)
        
        _, sgf_n = self.sgf.createSgf(copy.deepcopy(self.moves))
        board = GoBoard(sgf_n)
        
        return board.final_position()
    
    def main_loop(self, frame):
        self.frame = frame
        self.process_frame(frame)
        self.define_ordered_moves()
        print(len(self.moves), self.moves)
        
        # sgf_ = GoSgf('a', 'b', self.not_moves)
        # _, sgf_n = sgf_.createSgf()

        # if len(self.moves) > 0:
        #     _, sgf_n = self.sgf.update_sgf(self.all_moves[-1])
        # else:
        #     _, sgf_n = self.sgf.update_sgf([])
            
        _, sgf_n = self.sgf.createSgf(copy.deepcopy(self.moves))

        board = GoBoard(sgf_n)
        
        return board.final_position()
        

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
        horizontal_lines = removeDuplicates(horizontal_lines)
        
        vertical_lines = restore_missing_lines(vertical_lines)
        horizontal_lines = restore_missing_lines(horizontal_lines)

        vertical_lines = add_lines_in_the_edges(vertical_lines, "vertical")
        horizontal_lines = add_lines_in_the_edges(horizontal_lines, "horizontal")
        
        vertical_lines = removeDuplicates(vertical_lines)
        horizontal_lines = removeDuplicates(horizontal_lines)
        
        black_stones = get_key_points(self.results, 0, perspective_matrix)
        white_stones = get_key_points(self.results, 6, perspective_matrix)

        cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
        cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
                
        # img = np.copy(self.transformed_image)
        draw_lines(cluster_1, self.transformed_image)
        # img = np.copy(self.transformed_image)
        draw_lines(cluster_2, self.transformed_image)
        # imshow_(img)
        
        intersections = detect_intersections(cluster_1, cluster_2, self.transformed_image)
        
        
        
        if len(intersections) == 0:
            raise Exception(">>>>>No intersection were found!")
         
        self.update_game(white_stones, black_stones, intersections)
        
        
 


    
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
        
        draw_points(transformed_intersections, self.transformed_image, color=(0, 100, 0))
        
        
        self.map = create_board(transformed_intersections)
        
        # img = self.transformed_image.copy()
        # for key in self.map:
        #     draw_points(np.array([key]), img)
        #     cv2.putText(img, f'{self.map[key]}', np.array(key).astype(int), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.3, color=(0,0,255))
        # imshow_(img)
        
        
        self.old_state = copy.deepcopy(self.state)
        self.state = np.zeros((19, 19, 2))

        self.all_moves = []

        for stone in white_stones_transf:
            
            cv2.circle(self.transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
            
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            # cv2.circle(self.transformed_image, np.array(nearest_corner).astype(dtype=np.int32), 3, (0, 255, 0), 2)
            # print("W", stone, self.map[nearest_corner])
            self.state[self.map[nearest_corner][1], self.map[nearest_corner][0]] = 1
            self.all_moves.append(("W", (self.map[nearest_corner][0], 18 - self.map[nearest_corner][1])))
            
            cv2.line(self.transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
            # cv2.putText(self.transformed_image, f"{(self.map[nearest_corner])}", nearest_corner, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
            
                
        for stone in black_stones_transf:
            
            cv2.circle(self.transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
            
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            # cv2.circle(self.transformed_image, np.array(nearest_corner).astype(dtype=np.int32), 3, (0, 255, 0), 2)
            # print("B", stone, self.map[nearest_corner])
            self.state[self.map[nearest_corner][1], self.map[nearest_corner][0]] = 1000
            self.all_moves.append(("B", (self.map[nearest_corner][0], 18 - self.map[nearest_corner][1])))
            cv2.line(self.transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
            # cv2.putText(self.transformed_image, f"{(self.map[nearest_corner])}", nearest_corner, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
        
        # imshow_(self.transformed_image)
        self.STATES.append(self.state)
        
    
    def define_ordered_moves(self):
        difference = self.state - self.old_state
        pos_black = np.where(difference == 1000)
        pos_white = np.where(difference == 1)
        if len(pos_black[0]) + len(pos_white[0]) > 1:
            print("MORE THAN ONE STONE WAS ADDED")
            return
        if len(pos_black[0]) != 0:
            self.moves.append(('B', (pos_black[1][0], 18 - pos_black[0][0])))
            self.game.play()
            return
        if len(pos_white[0]) != 0: 
            self.moves.append(('W', (pos_white[1][0], 18 - pos_white[0][0])))
            return
        print("no moves detected")
        # game = sente.Game()
        

# #%%
# model = YOLO('model.pt')
# sgf = GoSgf('a', 'b')
# game = GoGame(model, sgf)

# frame = cv2.imread(f"img/{1}.jpg")
# imshow_(game.initialize_game(frame))

# print(game.moves)

# for i in range(2, 15):

#     frame = cv2.imread(f"img/{i}.jpg")
#     imshow_(game.main_loop(frame))
#     # print(game.moves)

# %%
