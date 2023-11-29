from processing import *
import copy
from mySgfCopy import GoSgf
import sente




class GoGame:

    def __init__(self, board_detect):
        self.history = []
        self.moves = []
        self.board_detect = board_detect
        self.current_state = []
        self.sgf = GoSgf("p1", "p2")
        # self.visual = GoVisual(self.sgf)
        self.game = sente.Game()


    def initialize_game(self, frame):
        # self.frame = frame
        # self.board_detect.process_frame(frame)
        # # self.moves = copy.deepcopy(self.all_moves)
        
        # _, sgf_n = self.sgf.createSgf(copy.deepcopy(self.moves))
        # board = GoBoard(sgf_n)
        
        # return board.final_position()
        return self.main_loop(frame)
    
    
    def main_loop(self, frame):
        self.frame = frame
        self.board_detect.process_frame(frame)
        self.define_new_move()
        print(len(self.moves), self.moves)
            
        _, sgf_n = self.sgf.createSgf(copy.deepcopy(self.moves))

        board = GoBoard(sgf_n)
        
        return board.final_position()
    
    def play_move(self, x, y, stone_color):
        try:
            self.game.play(x, y, sente.stone(stone_color))
        except sente.IllegalMoveException:
            raise Exception(f"A violation of go game rules has been found in position {x}, {y}")
            
    
    def define_new_move(self):
        detected_state = np.transpose(self.board_detect.get_state(), (1, 0, 2))
        current_state = self.game.numpy(["black_stones", "white_stones"])
        
        difference = detected_state - current_state
        black_stone_indices = np.argwhere(difference[:, :, 0] == 1)
        white_stone_indices = np.argwhere(difference[:, :, 1] == 1)
        
        print("black", np.argwhere(detected_state[:, :, 0] == 1))
        print("black", np.argwhere(current_state[:, :, 0] == 1))
        print("white", np.argwhere(detected_state[:, :, 1] == 1))
        print("white", np.argwhere(current_state[:, :, 1] == 1))
        
        if len(black_stone_indices) + len(white_stone_indices) > 1:
            print("MORE THAN ONE STONE WAS ADDED")
            return
        if len(black_stone_indices) != 0:
            self.play_move(black_stone_indices[0][0] + 1, black_stone_indices[0][1] + 1, 1) # sente.stone(1)/ 1 is black_stone
            self.moves.append(('B', (black_stone_indices[0][0], 18 - black_stone_indices[0][1])))
            return
        if len(white_stone_indices) != 0: 
            self.play_move(white_stone_indices[0][0] + 1, white_stone_indices[0][1] + 1, 2) # sente.stone(2)/ 2 is white_stone
            self.moves.append(('W', (white_stone_indices[0][0], 18 - white_stone_indices[0][1])))
            return
        print("no moves detected")

# %%