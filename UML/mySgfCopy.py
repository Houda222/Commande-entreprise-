#%%
from sgfmill import sgf
import numpy as np
import cv2
import sgf as sgf_
#%%

class GoBoard:
    """
    class Goboard: 
    creates a board given an sgf file provided by the GoSgf class
    can navigate through the game using methods such as previous or next
    """

    def __init__(self, sgf_url:str):
        """"
        Constructor method for GoBoard.

        Parameters:
        -----------
        sgf_url : str
            directory of the sgf file
        """
        with open(sgf_url, 'rb') as f:
            sgf_content = f.read()

        # Load an sgf file/ the game
        self.sgf_game = sgf.Sgf_game.from_bytes(sgf_content)
        
        #get the game size
        self.board_size = self.sgf_game.get_size()

        # Extract the game moves
        self.moves = []
        for i, node in enumerate(self.sgf_game.get_main_sequence(), 1):
            color, move = node.get_move()
            if color is not None and move is not None:
                row, col = move
                self.moves.append((row, col, color)) 
            # if is_stone_captured(sgf_url, i, move):
            #     print(f"Stone captured at {move}")

        # Get the number of moves 
        self.total_number_of_moves = len(self.moves)

        # Define the current number of moves initialized by the total number, and which we'll modify each time whne calling the previus or the next fucntion
        self.current_number_of_moves = self.total_number_of_moves
        
       
    def drawBoard(self, number_of_moves_to_show : int):
        """
        Draw the board up to a certain number of moves

        Parameters:
        -----------
        number_of_moves_to_show : int
            Define moves we want to plot on the board

        Returns:
        --------
        numpy array
            The resulted board 
        """
        square_size = 30
        circle_radius = 12
        
        #set up the board's background
        board =np.full(((self.board_size+1)*square_size, (self.board_size+1)*square_size, 3), (69, 166, 245), dtype=np.uint8)
        board2 = np.zeros((self.board_size, self.board_size))

        #extract the moves we wanna show
        extracted_moves = self.moves[:number_of_moves_to_show]
        
        # Draw lines for the board grid
        
        # for i in range(board_size):
        #     ax.plot([i, i], [0, board_size - 1], color='k', linewidth = 0.7)
        #     ax.plot([0, board_size - 1], [i, i], color='k', linewidth = 0.7)
        
        for i in range(1, self.board_size+1):
            # Vertical lines and letters
            cv2.line(board, (square_size*i, square_size), (square_size*i, square_size*(self.board_size)), (0, 0, 0), thickness=1)
            #plt.text(i, -0.8, chr(97 + i), fontsize=8, color='black')    
            cv2.putText(board, str(i), (square_size*i, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
            # Horizontal lines and letters
            cv2.line(board, (square_size, square_size*i), (square_size*(self.board_size), square_size*i), (0, 0, 0), thickness=1)
            #plt.text(-0.8, i, chr(97 + i), fontsize=8, color='black')  
            cv2.putText(board, str(i), (5, square_size*i), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
        # Draw stones
        for move in extracted_moves:
            row, col, color = move
            if board2[row, col] == 0:
                stone_color = (0, 0, 0) if color == 'b' else (255, 255, 255)
                board2[row, col] = 1
                cv2.circle(board, ((col+1)*square_size, (row+1)*square_size), circle_radius, color=(66, 66, 66), thickness=2) # draw the edge
                cv2.circle(board, ((col+1)*square_size, (row+1)*square_size), circle_radius, color=stone_color, thickness=-1) # draw the stone
        
        #setting the contour of the last move to a different color
        if len(extracted_moves) != 0:
            last_move = extracted_moves[-1]           
            stone_color = (0,0,0) if last_move[2] == 'b' else (255, 255, 255)
            cv2.circle(board, ((last_move[1]+1)*square_size, (last_move[0] + 1)*square_size), circle_radius, color=(0,0,255), thickness=2) 
            cv2.circle(board, ((last_move[1]+1)*square_size, (last_move[0] + 1)*square_size), circle_radius, color=stone_color, thickness=-1) 

        return board

    
    
    def initial_position(self):
        """
        Display the initial position with the first move

        Returns:
        --------
        numpy array
            The resulted board drawn with only the first played move
        """
        self.current_number_of_moves = 1
        return self.drawBoard(1)

    def final_position(self):
        """
        Display the final position 

        Returns:
        --------
        numpy array
            The resulted board drawn with all the played moves 
        """
        self.current_number_of_moves = self.total_number_of_moves
        return self.drawBoard(self.total_number_of_moves)

    def current_position(self):
        """
        Display the current position

        Returns:
        --------
        numpy array
            The resulted board drawn with all the played moves up to the current instant
        """
        return self.drawBoard(self.current_number_of_moves)

    def current_turn(self):
        """
        Display whose turn to play

        Returns:
        --------
        string
            The color of the current turn
        """
        turn = self.moves[self.current_number_of_moves - 1][2]
        if turn == 'b':
            return 'White' 
        elif turn == 'w' or self.current_number_of_moves == 0:
            return 'black'
        
    def previous(self):
        """
        Display the previous position

        Returns:
        --------
        numpy array
            The board one move before the displayed position
        """
        if self.current_number_of_moves > 1:
            self.current_number_of_moves -= 1
            return self.drawBoard(self.current_number_of_moves)

    def next(self):
        """
        Display the next position

        Returns:
        --------
        numpy array
            The board one move after the displayed position
        """
        if self.current_number_of_moves < self.total_number_of_moves:
            self.current_number_of_moves += 1
            return self.drawBoard(self.current_number_of_moves)

    

            
class GoSgf:
    """
    class GoSgf: 
    creates an sgf file given the list of moves (stones and their positions) extracted from the image recognition part
    """

    def __init__(self, black:str, white:str, tournament:str=None, date:str=None):
        """"
        Constructor method for GoSgf.

        Parameters:
        -----------
        black : str
            name of the black player
        white : str
            name of the white player
        moves : list
            a list of the played moves
        tournament : str
            tournament in which the game was played, can be attributed "training"
        date : str
            date in which the game was played
        """
        # define the game information
        self.black = black
        self.white = white
        self.board_size = (19,19)
        self.tournament = tournament

        self.game_info = {
            "EV" : tournament,
    	    "RO" : "1",
            #"GM" : "1", #game type, 1 for go
            "PB" : black,
            "PW" : white,
            #"SZ" : f"{self.board_size[0]}",
            "KM" : "6.5", #komi
            #"RU" : "Japanese" #rules used
            "DT" : date
        }

        #get the moves we collected from board recognition
        self.moves = []

    def update_sgf(self, move):
        """
        Add a move to the game and update the SGF file.

        Parameters:
        -----------
        player : str
            Player color ('B' for black, 'W' for white)
        position : tuple
            Tuple representing the (x, y) coordinates of the move
        """
        player, position = move
        self.moves.append((player, position))
        sgf_content = self.assembleSgf()

        sgf_filename = f"{self.black}_{self.white}.sgf"
        with open(sgf_filename, "w") as sgf_file:
            # Write game information
            sgf_file.write("(; \n")
            for key, value in self.game_info.items():
                sgf_file.write(f"{key}[{value}]\n")
            sgf_file.write("\n")

            # Write stone positions
            sgf_file.write(sgf_content)

            # End the SGF file
            sgf_file.write(")\n")
        
        return sgf_file, sgf_filename


    # ... (your existing methods)
    
    #convert a move to SGF format
    def add_to_sgf(self, move):
        """
        Add a move to the sgf file

        Returns:
        --------
        str
            one move composed of the player, the letter corresponding to the x coordinate, and the letter corresponding to the y coordinate
        """
        player, position = move
        x, y = position 
        sgf_x = chr(ord('a') + x)
        sgf_y = chr(ord('a') + y)
        return f";{player}[{sgf_x}{sgf_y}]"
        
    #convert the sgf file 
    def assembleSgf(self):
        """
        Put together and write the sgf file and save it

        Returns:
        --------
        sgf
            the sgf file of the game

        str
            The name of the sgf file
        """
        sgf_ = ''.join([self.add_to_sgf(move) for move in self.moves])
        return sgf_
    

    def createSgf(self, moves):
        """
        Create and Write the sgf file

        Returns:
        --------
        sgf_file : sgf
            the sgf file of the game
        sgf_filename : str
            name of the sgf file
        """
        self.moves = moves
        sgf_moves = self.assembleSgf()
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
    
    

    
# def is_stone_captured(move):
    
#         if coordinate in node.properties.get('B', []) or coordinate in node.properties.get('W', []):
#             # Check if the stone has liberties
#             liberties = get_liberties(node, coordinate)
#             return len(liberties) == 0

#         return False
# def is_stone_captured(color, move):
#     """
#     Check if a stone is captured after a given move.

#     Parameters:
#     - game: The game object.
#     - move: The move to be checked.

#     Returns:
#     - True if a stone is captured, False otherwise.
#     """
#     # Extract the color and coordinates of the move
#     (row, col) = move

#     # Get the opponent's color
#     opponent_color = 'b' if color == 'w' else 'w'

#     # Check if any adjacent opponent stones are captured
#     captured_stones = []
#     for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
#         new_row, new_col = row + dr, col + dc

#         if 1 <= new_row <= self.board_size and 1 <= new_col <= self.board_size:
#             neighbor_color, _ = game.get_board().get((new_row, new_col), (None, None))

#             # If the neighbor is an opponent's stone and it's part of a captured group,
#             # add it to the list of captured stones
#             if neighbor_color == opponent_color and game.get_group((new_row, new_col)) is None:
#                 captured_stones.extend(game.get_captured_stones((new_row, new_col)))

#     return len(captured_stones) > 0

# Usage example:
# Assuming that you have a Game object called 'game' and a move tuple called 'move'
# is_stone_captured(game, move)

# def get_liberties(node, coordinate):
#     # Extract the board state from the SGF node
#     board_size = int(node.properties.get('SZ', ['19'])[0])  # Assuming a default size of 19x19
#     board = [[' ' for _ in range(board_size)] for _ in range(board_size)]

#     # Fill in the stones from the SGF node
#     for color, positions in [('B', node.properties.get('B', [])), ('W', node.properties.get('W', []))]:
#         for pos in positions:
#             row, col = sgf_coordinates_to_indices(pos)
#             board[row][col] = color

#     # Find the group to which the stone belongs
#     group = find_group(board, coordinate)

#     # Get liberties of the group
#     liberties = set()
#     for stone in group:
#         liberties.update(get_adjacent_empty_positions(board, stone))

#     return liberties

# def sgf_coordinates_to_indices(sgf_coordinate):
#     col = ord(sgf_coordinate[0].upper()) - ord('A')
#     row = int(sgf_coordinate[1:]) - 1
#     return row, col

# def find_group(board, start_position):
#     color = board[start_position[0]][start_position[1]]
#     group = set()
#     visited = set()

#     def dfs(position):
#         if position in visited or board[position[0]][position[1]] != color:
#             return
#         visited.add(position)
#         group.add(position)

#         for neighbor in get_adjacent_positions(position, board_size=len(board)):
#             dfs(neighbor)

#     dfs(start_position)
#     return group

# def get_adjacent_positions(position, board_size):
#     row, col = position
#     directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#     adjacent_positions = [(row + dr, col + dc) for dr, dc in directions]

#     return [(r, c) for r, c in adjacent_positions if 0 <= r < board_size and 0 <= c < board_size]

# def get_adjacent_empty_positions(board, position):
#     empty_positions = []
#     for neighbor in get_adjacent_positions(position, board_size=len(board)):
#         if board[neighbor[0]][neighbor[1]] == ' ':
#             empty_positions.append(neighbor)
#     return empty_positions


# #%%
# ###example
# moves = [('B', (9, 9)), ('W', (14, 18)), ('B', (14, 17)), ('W', (13, 17)), ('B', (3, 15)), ('W', (6, 15)), ('B', (14, 15)), ('W', (0, 14)), ('B', (14, 13)), ('W', (13, 13)), ('B', (0, 10)), ('W', (9, 9)), ('B', (10, 9)), ('W', (11, 9))]
# fichier = GoSgf("ex", "ex")
# sgf_file, sgf_filename = fichier.update_sgf(moves[-1])
# #%%
# board = GoBoard(sgf_filename)
# res = board.final_position()
# cv2.imshow("result", res)
# cv2.waitKey(0)