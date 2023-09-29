#%%
from sgfmill import sgf, sgf_moves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Gosgf:
    def __init__(self, sgf_url:str):
        with open(sgf_url, 'rb') as f:
            sgf_content = f.read()
        
        # Loading an sgf file/ the game
        self.sgf_game = sgf.Sgf_game.from_bytes(sgf_content)
        
        # Extract the game moves
        self.board_size = self.sgf_game.get_size()

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
        self.current_position = self.moves
        
       
       
    # Draw the board, the board is drawn for the current/final position by default
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

            
    
#using the class
game = Gosgf('4.sgf')
game.final_position()
# %%
