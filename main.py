# %%
import threading
import copy
import traceback
from ultralytics import YOLO
import cv2
from processing import show_board
from GoGame import *
# %%
def processing_thread():
    global ProcessFrame, Process
    while Process:
        if not ProcessFrame is None:
            try:
                
                ############ WA SMA3NI MZZZZN DB.  game_plot HYA LA VARIABLE LLI FIHA L'IMAGE DESSINé
                ############ B LE CODE DYAL HOUDA;
                ############ O sgf_filename HOWA LE NOM DYAL LE FICHER SGF LLI T ENREGISTRA 
                ############ QUI CORRESPOND A game_plot
                game_plot = game.process_frame(frame)
                # game_plot, sgf_filename = show_board(model, ProcessFrame)
                cv2.imshow("master", game_plot)
                
            except OverflowError as e:
                print(f"Overflow Error: {e}")
                
            except Exception as e:
                print('empty frame', type(e), e.args, e)
                traceback.print_exc()
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Process = False
            break  # Break the loop if 'q' is pressed


model = YOLO('model.pt')
game = GoGame(model)

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
# %%

import threading
import copy
import traceback
from ultralytics import YOLO
import cv2
from processing import *
from GoGame import *



model = YOLO('model.pt')
game = GoGame(model)
frame = cv2.imread(f"img/{1}.jpg")
model_results = model(frame)

input_points = get_corners(model_results)

output_edge = 600
output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)

empty_intersections = model_results[0].boxes.xywh[model_results[0].boxes.cls == 3][:,[0, 1]]
empty_corner = model_results[0].boxes.xywh[model_results[0].boxes.cls == 4][:,[0, 1]]
empty_edge = model_results[0].boxes.xywh[model_results[0].boxes.cls == 5][:,[0, 1]]


if not empty_intersections is None:
    if len(empty_intersections) != 0:
        empty_intersections = np.array(empty_intersections[:, [0, 1]])
        empty_intersections = cv2.perspectiveTransform(empty_intersections.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

if not empty_corner is None:
    if len(empty_corner) != 0:
        empty_corner = np.array(empty_corner[:, [0, 1]])
        empty_corner = cv2.perspectiveTransform(empty_corner.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

if not empty_edge is None:
    if len(empty_edge) != 0:
        empty_edge = np.array(empty_edge[:, [0, 1]])
        empty_edge = cv2.perspectiveTransform(empty_edge.reshape((1, -1, 2)), perspective_matrix).reshape((-1, 2))

all_intersections = np.concatenate((empty_intersections, empty_corner, empty_edge), axis=0)

all_intersections = all_intersections[(all_intersections[:, 0:2] >= 0).all(axis=1) & (all_intersections[:, 0:2] <= 600).all(axis=1)]

# %%
# remove a given number of random elements
num_elements_to_remove = 2

# Generate random indices to remove
indices_to_remove = np.random.choice(all_intersections.shape[0], num_elements_to_remove, replace=False)

# Remove the selected indices
all_intersections_test = np.delete(all_intersections, indices_to_remove, axis=0)
# %%
