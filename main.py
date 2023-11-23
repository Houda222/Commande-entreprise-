import threading
import copy
import traceback
from ultralytics import YOLO
import cv2
from processing import show_board
from GoGame import *


def processing_thread():
    global ProcessFrame, Process
    
    initialized = False
    while Process:
        if not ProcessFrame is None:
            try:
                if not initialized:
                    game_plot = game.initialize_game(frame)
                    initialized = True
                else:
                    
                ############ WA SMA3NI MZZZZN DB.  game_plot HYA LA VARIABLE LLI FIHA L'IMAGE DESSINé
                ############ B LE CODE DYAL HOUDA;
                ############ O sgf_filename HOWA LE NOM DYAL LE FICHER SGF LLI T ENREGISTRA 
                ############ QUI CORRESPOND A game_plot
                    game_plot = game.main_loop(frame)
                # game_plot, sgf_filename = show_board(model, ProcessFrame)
                # cv2.imshow("master", game_plot)
                cv2.imshow("annotated", game.annotated_frame)
                # cv2.imshow("transformed", game.transformed_image)
                
            except OverflowError as e:
                print(f"Overflow Error: {e}")
                
            except Exception as e:
                print('empty frame', type(e), e.args, e)
                traceback.print_exc()
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Process = False
            break  # Break the loop if 'q' is pressed


model = YOLO('model.pt')
sgf = GoSgf('a', 'b')
game = GoGame(model, sgf)

ProcessFrame = None
Process = True

process_thread = threading.Thread(target=processing_thread, args=())
process_thread.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    ########################## frame HYA LA VARIABLE LLI FIHA CHAQUE IMAGE DYAL STREAM
    ########################## YA3NI LE FLUX DE VIDEO QUI DOIT ETRE STREAMé
    ProcessFrame = copy.deepcopy(frame)
    
    cv2.imshow('Video Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        Process = False
        break 

cap.release()
cv2.destroyAllWindows()