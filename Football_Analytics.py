import cv2
import numpy as np
import PySimpleGUI as sg
from ultralytics import YOLO
from GUI import layout,window_width,window_height
from Definitions import Team, get_main_colors, team_recognizer, resize_frame, field_lines
from Set_up import set_up


#starting variables are initialized
video_path = "video/Example1.mov"
video_path = "video/Example2.mp4"
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(video_path)
array_col = []
team_colors = list()
ball_out = False
passage = False
last_pos = 2
new_pos = 2
bc = '#323232' # background color
side = 250
last_ball_coords = []
adjust_indices = [0, 0, 0, 0]
window = sg.Window('Football Analytics', layout, finalize=True, resizable=True, background_color=bc)
window.Size = (window_width, window_height)
team_colors = set_up(array_col, window, cap, model)
team1 = Team(team_colors[0])
team2 = Team(team_colors[1])


# Main loop
while True:
    # Read the frame
    ret, frame = cap.read()

    # If the frame was not read correctly, break from the loop
    if not ret:
        break

    crop_frame = frame[::]
    highest_points, rightmost_points, lowest_points, leftmost_points = field_lines(frame)
    ball_found = False
    result = model.predict(crop_frame, verbose = False)[0]
    n_team1 = 0
    n_team2 = 0
    possessors = []

    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        prob = round(box.conf[0].item(), 2)
        cords = box.xyxy[0].tolist()
        x,y,w,h = [round(x) for x in cords]
        
        if class_id == "person" and prob > 0.50:
                rectangular_portion = crop_frame[int((y+h)//2.02-(0.1*(h-y))):int((y+h)//2.02+(0.1*(h-y))), int((x+w)//2-(0.1*(w-x))):int((x+w)//2+(0.1*(w-x)))]
                main_color = get_main_colors(rectangular_portion)
                label_id = team_recognizer(team_colors[0], team_colors[1], main_color)
                if label_id != None:
                    label = team_colors[label_id]
                    if label_id == 0:
                        n_team1 += 1
                        team1.player.append((x,y,w,h))
                    else:
                        n_team2 += 1
                        team2.player.append((x,y,w,h))
                
                    cv2.rectangle(frame,(x,y), (w,h), label, 3)
           
        # Ball detection, we don't use the accuracy because it is not very accurate and there is the risk
        # of never finding the ball if the accuracy is too high.
        if class_id == "sports ball":
            ball_found = True
            last_ball_coords = cords
            cv2.rectangle(frame,(x , y ),   (w ,h ), (0, 0, 255), 2)


    if ball_found:
        # Save the ball's coordinates
        xb,yb,wb,hb = [round(x) for x in last_ball_coords]
        ball_center = np.array(((xb+wb)//2, (yb+hb)//2))
        distance_team1 = float('inf')
        distance_team2 = float('inf')


        # We iterate over the players of each team and we find the smallest Euclidean distance between the ball and the players' feet.
        for player in team1.player:
            xp,yp,wp,hp = [round(x) for x in player]
            player1_feet_center = np.array(((xp+wp)//2, yp+int(0.9*(hp-yp))))
            distance_team1 = min(distance_team1, np.linalg.norm(ball_center - player1_feet_center, ord=2))

        for player in team2.player:
            xp,yp,wp,hp = [round(x) for x in player]
            player2_feet_center = np.array(((xp+wp)//2, yp+int(0.9*(hp-yp))))
            distance_team2 = min(distance_team2, np.linalg.norm(ball_center - player2_feet_center, ord=2))
        

        # We update the ball possession based on the distance
        if distance_team1 < distance_team2 and distance_team1 < 25:
           team1.ball_possession += 1
           possessors.append('1')
           new_pos = 0
        elif distance_team2 < distance_team1 and distance_team2 < 25:
           team2.ball_possession += 1
           possessors.append('2')
           new_pos = 1


        # We update the tackles and the passages
        if new_pos != last_pos and last_pos == 1:
            team2.tackles += 1
        elif new_pos != last_pos and last_pos == 0:
            team1.tackles += 1

        if new_pos == 0 and distance_team1 > 30: 
            passage = True
        elif new_pos == 1 and distance_team2 > 30:
            passage = True


        # We update the ball position
        if not passage:
            ball_position = ball_center
            #print(ball_position)
            possession = new_pos

        # We update the passages and the tackles
        if passage and new_pos == 0 == possession and distance_team1 < 30 and ball_position[0] != ball_center[0] and ball_position[1] != ball_center[1] and np.abs(ball_position[0] - ball_center[0]) > 30 and np.abs(ball_position[1] - ball_center[1]) > 30:
            team1.passages += 1
            passage = False
        elif passage and new_pos == 1 == possession and distance_team2 < 30 and ball_position[0] != ball_center[0] and ball_position[1] != ball_center[1] and np.abs(ball_position[0] - ball_center[0]) > 30 and np.abs(ball_position[1] - ball_center[1]) > 30:
            team2.passages += 1
            passage = False
        elif passage and new_pos != possession and ball_position[0] != ball_center[0] and ball_position[1] != ball_center[1] and np.abs(ball_position[0] - ball_center[0]) > 20 and np.abs(ball_position[1] - ball_center[1]) > 20:
            if new_pos == 0:
                team1.tackles += 1
            else:
                team2.tackles += 1
            possession = new_pos
            passage = False

        # We check if the ball is out of the field.
        height, width = frame.shape[:2]
        # print(height, width)
        border_width = int(width * 0.1)
        border_height = int(height * 0.1)
        border_top = 0 + border_height
        border_bottom = height - border_height
        border_left = 0 + border_width
        border_right = width - border_width
        if highest_points:
            is_inside_field = (
                ball_center[0] > border_left and
                ball_center[0] < border_right and
                ball_center[1] > border_top and
                ball_center[1] < border_bottom
            )

            # print(f"Ball  center: {ball_center}, borders: {[border_top,  border_right, border_bottom, border_left]}")
            if not is_inside_field and (ball_center[1] < all([i[1] for i in highest_points]) or ball_center[1] > all([i[1] for i in lowest_points]) or ball_center[0] > all([i[0] for i in rightmost_points]) or ball_center[0] < all([i[0] for i in leftmost_points])):
                try:
                    if possessors[-1] == '1': team1.outs += 1 
                    if possessors[-1] == '2': team2.outs += 1 
                    ball_out = True
                except:
                    pass
            else:
                ball_out = False
        else:
            ball_out = False

        last_pos = new_pos

    # We update the statistics
    # First we check if the sum of the ball possessions is not 0, otherwise we would have a division by 0.
    if team1.ball_possession+team2.ball_possession != 0:
            t1_bp = round(team1.ball_possession/(team1.ball_possession+team2.ball_possession)*100, 1)
            t2_bp = round(team2.ball_possession/(team1.ball_possession+team2.ball_possession)*100, 1)
    else:
            t1_bp = 0
            t2_bp = 0



    # Update the PySimpleGUI window
    resized_frame = resize_frame(frame, int(0.75*window_width), int(window_height))
    window['-image-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
    window['-playersi-'].update(f'{n_team1+n_team2}')
    window['-players_team1i-'].update(f'{n_team1}')
    window['-players_team2i-'].update(f'{n_team2}')
    window['-ball_possession1i-'].update(f'{t1_bp}%')
    window['-ball_possession2i-'].update(f'{t2_bp}%')
    window['-tackles_ti-'].update(f'{team1.tackles+team2.tackles}')
    window['-tackles_team1i-'].update(f'{team1.tackles}')
    window['-tackles_team2i-'].update(f'{team2.tackles}')
    window['-outs_team1i-'].update(f"{team1.outs}")
    window['-outs_team2i-'].update(f"{team2.outs}")
    window['-passages_team1i-'].update(f'{team1.passages}')
    window['-passages_team2i-'].update(f'{team2.passages}')
    

    event, values = window.read(timeout=20) #timeout=20 is the time in ms between each frame in # type: ignore
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break

# Release the video file and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the PySimpleGUI window
window.close()