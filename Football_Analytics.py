import cv2
import numpy as np
import base64
import threading
import torch
from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO
from Definitions import Team, get_main_colors, team_recognizer, resize_frame, field_lines, bgr_to_hex
from Set_up import set_up

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
video_path = "video/match.mp4"


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@app.route('/')
def index():
    return render_template('index.html')


def process_video():
    device = get_device()
    print(f"Using device: {device}")
    model = YOLO("yolov8m.pt")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    array_col = []
    print("Running setup — sampling frames to identify team colors...")
    team_colors = set_up(array_col, cap, model)
    print(f"Team colors identified: {team_colors}")

    team1 = Team(team_colors[0])
    team2 = Team(team_colors[1])

    # Emit team colors to frontend
    socketio.emit('team_colors', {
        'team1': bgr_to_hex(team_colors[0]),
        'team2': bgr_to_hex(team_colors[1])
    })

    ball_out = False
    passage = False
    last_pos = 2
    new_pos = 2
    last_ball_coords = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            socketio.emit('video_ended')
            break

        crop_frame = frame[::]
        highest_points, rightmost_points, lowest_points, leftmost_points = field_lines(frame)
        ball_found = False
        result = model.predict(crop_frame, verbose=False)[0]
        n_team1 = 0
        n_team2 = 0
        possessors = []

        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            prob = round(box.conf[0].item(), 2)
            cords = box.xyxy[0].tolist()
            x, y, w, h = [round(x) for x in cords]

            if class_id == "person" and prob > 0.50:
                rectangular_portion = crop_frame[int((y+h)//2.02-(0.1*(h-y))):int((y+h)//2.02+(0.1*(h-y))), int((x+w)//2-(0.1*(w-x))):int((x+w)//2+(0.1*(w-x)))]
                main_color = get_main_colors(rectangular_portion)
                label_id = team_recognizer(team_colors[0], team_colors[1], main_color)
                if label_id is not None:
                    label = team_colors[label_id]
                    if label_id == 0:
                        n_team1 += 1
                        team1.player.append((x, y, w, h))
                    else:
                        n_team2 += 1
                        team2.player.append((x, y, w, h))

                    cv2.rectangle(frame, (x, y), (w, h), label, 3)

            if class_id == "sports ball":
                ball_found = True
                last_ball_coords = cords
                cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

        if ball_found:
            xb, yb, wb, hb = [round(x) for x in last_ball_coords]
            ball_center = np.array(((xb+wb)//2, (yb+hb)//2))
            distance_team1 = float('inf')
            distance_team2 = float('inf')

            for player in team1.player:
                xp, yp, wp, hp = [round(x) for x in player]
                player1_feet_center = np.array(((xp+wp)//2, yp+int(0.9*(hp-yp))))
                distance_team1 = min(distance_team1, np.linalg.norm(ball_center - player1_feet_center, ord=2))

            for player in team2.player:
                xp, yp, wp, hp = [round(x) for x in player]
                player2_feet_center = np.array(((xp+wp)//2, yp+int(0.9*(hp-yp))))
                distance_team2 = min(distance_team2, np.linalg.norm(ball_center - player2_feet_center, ord=2))

            if distance_team1 < distance_team2 and distance_team1 < 25:
                team1.ball_possession += 1
                possessors.append('1')
                new_pos = 0
            elif distance_team2 < distance_team1 and distance_team2 < 25:
                team2.ball_possession += 1
                possessors.append('2')
                new_pos = 1

            if new_pos != last_pos and last_pos == 1:
                team2.tackles += 1
            elif new_pos != last_pos and last_pos == 0:
                team1.tackles += 1

            if new_pos == 0 and distance_team1 > 30:
                passage = True
            elif new_pos == 1 and distance_team2 > 30:
                passage = True

            if not passage:
                ball_position = ball_center
                possession = new_pos

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

            height, width = frame.shape[:2]
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

        # Compute stats
        if team1.ball_possession + team2.ball_possession != 0:
            t1_bp = round(team1.ball_possession / (team1.ball_possession + team2.ball_possession) * 100, 1)
            t2_bp = round(team2.ball_possession / (team1.ball_possession + team2.ball_possession) * 100, 1)
        else:
            t1_bp = 0
            t2_bp = 0

        # Encode frame as JPEG base64
        resized_frame = resize_frame(frame, 960, 540)
        _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Emit frame and stats
        socketio.emit('frame', {
            'image': frame_b64,
            'stats': {
                'players_total': n_team1 + n_team2,
                'players_team1': n_team1,
                'players_team2': n_team2,
                'possession_team1': t1_bp,
                'possession_team2': t2_bp,
                'tackles_total': team1.tackles + team2.tackles,
                'tackles_team1': team1.tackles,
                'tackles_team2': team2.tackles,
                'outs_team1': team1.outs,
                'outs_team2': team2.outs,
                'passages_team1': team1.passages,
                'passages_team2': team2.passages,
            }
        })

        # Reset player lists for next frame
        team1.player = []
        team2.player = []

        socketio.sleep(frame_delay)

    cap.release()


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    threading.Thread(target=process_video, daemon=True).start()


if __name__ == '__main__':
    print("Starting Football Analytics server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
