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

#print(moves)

# Draw the board 
def board():
    board = np.zeros((board_size, board_size))
    fig, ax = plt.subplots(figsize=(8, 8))
    
    #set up the board's background
    background = patches.Rectangle((-1,-1), board_size + 1, board_size + 1, facecolor='#EEAD0E', fill = True, edgecolor='black')
    ax.add_patch(background)

    # Draw lines for the board grid
    
    # for i in range(board_size):
    #     ax.plot([i, i], [0, board_size - 1], color='k', linewidth = 0.7)
    #     ax.plot([0, board_size - 1], [i, i], color='k', linewidth = 0.7)
    
    for i in range(board_size):
        # Vertical lines
        ax.add_patch(patches.Rectangle((i - 0.01, -0.01), 0.02, board_size + 0.02 -1, color='k'))
        # Horizontal lines
        ax.add_patch(patches.Rectangle((-0.01, i - 0.01), board_size + 0.02 - 1, 0.02, color='k'))

    ax.set_xlim(-1, board_size)
    ax.set_ylim(-1, board_size)

    # Draw stones
    for move in moves:
        row, col, color = move
        if board[row, col] == 0:
            stone_color = 'black' if color == 'b' else 'white'
            board[row, col] = 1
            ax.add_patch(plt.Circle((col, board_size - row - 1), 0.4, facecolor=stone_color, fill = True, edgecolor = 'black'))

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

board()




