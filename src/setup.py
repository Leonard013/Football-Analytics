import random
from sklearn.cluster import KMeans
import numpy as np
import cv2
from src.detection import color_picker, isframe


def set_up(array_col, cap, model):

    while len(array_col) < 200:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_index = random.randint(0, frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(random_frame_index))
        ret, frame = cap.read()
        result = model.predict(frame, verbose=False)[0]

        for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                prob = round(box.conf[0].item(), 2)
                cords = box.xyxy[0].tolist()
                x, y, w, h = [round(x) for x in cords]
                if class_id == "person" and prob > 0.50 and isframe(frame):
                    array_col.append(color_picker(x, y, w, h, frame))

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(np.array(array_col).reshape(-1, 3))
    team_colors = kmeans.cluster_centers_
    team_colors = [(int(r), int(g), int(b)) for r, g, b in team_colors]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return team_colors
