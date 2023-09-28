#%%
from sgfmill import sgf, sgf_moves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Loading an sgf file/ the game
with open('4.sgf', 'rb') as f:
    sgf_content = f.read()

sgf_game = sgf.Sgf_game.from_bytes(sgf_content)


# Extract the game moves
board_size = sgf_game.get_size()
moves = []


for node in sgf_game.get_main_sequence():
    color, move = node.get_move()
    if color is not None and move is not None:
        row, col = move
        moves.append((row, col, color)) 

#get the number of moves 
total_number_of_moves = len(moves)

#define the current number of moves initialized by the total number, and which we'll modify each time whne calling the previus or the next fucntion
current_number_of_moves = total_number_of_moves
current_position = moves


# Draw the board 
def board(number_of_moves_to_show = total_number_of_moves):
    board = np.zeros((board_size, board_size))
    fig, ax = plt.subplots(figsize=(8, 8))

    #extract the moves we wanna show
    extracted_moves = moves[:number_of_moves_to_show]

    #set up the board's background
    background = patches.Rectangle((-1,-1), board_size + 1, board_size + 1, facecolor='#EEAD0E', fill = True, edgecolor='black')
    ax.add_patch(background)

    # Draw lines for the board grid
    
    # for i in range(board_size):
    #     ax.plot([i, i], [0, board_size - 1], color='k', linewidth = 0.7)
    #     ax.plot([0, board_size - 1], [i, i], color='k', linewidth = 0.7)
    
    for i in range(board_size):
        # Vertical lines and letters
        ax.add_patch(patches.Rectangle((i - 0.01, -0.01), 0.02, board_size + 0.02 -1, color='k'))
        plt.text(i, -0.8, chr(97 + i), fontsize=8, color='black')       
        # Horizontal lines and letters
        ax.add_patch(patches.Rectangle((-0.01, i - 0.01), board_size + 0.02 - 1, 0.02, color='k'))
        plt.text(-0.8, i, chr(97 + i), fontsize=8, color='black')       


    # Set axis limits to include the entire grid
    ax.set_xlim(-1, board_size)
    ax.set_ylim(-1, board_size)

    # Draw stones
    for move in extracted_moves:
        row, col, color = move
        if board[row, col] == 0:
            stone_color = 'black' if color == 'b' else 'white'
            board[row, col] = 1
            ax.add_patch(plt.Circle((col, board_size - row - 1), 0.4, facecolor=stone_color, fill = True, edgecolor = 'black'))
    
    #setting the contour of the last move to a different color
    last_move = extracted_moves[-1]           
    stone_color = 'black' if last_move[2] == 'b' else 'white'
    ax.add_patch(patches.Circle((last_move[1], board_size - last_move[0] - 1), 0.4, facecolor=stone_color, fill = True, edgecolor = 'red'))

    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


#access to previous move
def previous():
    global current_number_of_moves
    if current_number_of_moves > 1:
        current_number_of_moves -= 1
        board(current_number_of_moves)


#access to next move
def next():
    global current_number_of_moves
    if current_number_of_moves < total_number_of_moves:
        current_number_of_moves += 1
        board(current_number_of_moves)


#display the initial position with the first move
def initial_position():
    global current_number_of_moves
    current_number_of_moves = 1
    board(1)


#display the final position
def final_position():
    global current_number_of_moves
    current_number_of_moves = total_number_of_moves
    board()


#display the current position
def current_position():
    global current_number_of_moves
    board(current_number_of_moves)


#display whose turn to play
def current_turn():
    turn = moves[current_number_of_moves - 1][2]
    if turn == 'b':
        return 'White' 
    elif turn == 'w' or current_number_of_moves == 0:
        return 'black'
    



