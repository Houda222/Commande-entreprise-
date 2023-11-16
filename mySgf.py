from sgfmill import sgf
import numpy as np
import cv2


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
        for node in self.sgf_game.get_main_sequence():
            color, move = node.get_move()
            if color is not None and move is not None:
                row, col = move
                self.moves.append((row, col, color)) 

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

    def __init__(self, black:str, white:str, moves:list, tournament:str=None, date:str=None):
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
        self.moves = moves

    def createSgf(self):
        """
        Create and Write the sgf file

        Returns:
        --------
        sgf_file : sgf
            the sgf file of the game
        sgf_filename : str
            name of the sgf file
        """
        
        #convert a move to SGF format
        def add_to_sgf(move):
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
        def assembleSgf():
            """
            Put together and write the sgf file and save it

            Returns:
            --------
            sgf
                the sgf file of the game

            str
                The name of the sgf file
            """
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
    # def manually_add_a_move(self, move):
    #     """
    #     Add a move to the sgf file

    #     Returns:
    #     --------
    #     str
    #         one move composed of the player, the x coordinate, and the y coordinate
    #     """
    #     self.moves.append(move)
    #     for move in self.moves:
    #         player, position = move
    #         x, y = position 
    #         sgf_x = chr(ord('a') + x)
    #         sgf_y = chr(ord('a') + y)
    #         to_add =  f";{player}[{sgf_x}{sgf_y}]"
    #         sgf_ = ''.join([add_to_sgf(move) for move in self.moves])

    # def delete_a_move(self, move):
    
        



####example
# moves = [('B', (9, 9)), ('W', (14, 18)), ('B', (14, 17)), ('W', (13, 17)), ('B', (3, 15)), ('W', (6, 15)), ('B', (14, 15)), ('W', (0, 14)), ('B', (14, 13)), ('W', (13, 13)), ('B', (0, 10)), ('W', (9, 9)), ('B', (10, 9)), ('W', (11, 9))]
# fichier = GoSgf("ex", "ex", moves)
# sgf_file, sgf_filename = fichier.createSgf()

# board = GoBoard(sgf_filename)
# res = board.final_position()
# cv2.imshow("result", res)
# cv2.waitKey(0)
