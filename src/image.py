import cv2
import numpy as np


# It returns the hex code of a color given its BGR code
def bgr_to_hex(bgr):
    b, g, r = bgr
    hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    return hex_color

# It returns the new width and height of the frame given the maximum width and height to fit the video in the screen
def resize_frame(frame, max_width, max_height):

    width, height = frame.shape[1], frame.shape[0]
    width_scale = max_width / width
    height_scale = max_height / height
    scale = min(width_scale, height_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    background = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    x = (max_width - new_width) // 2
    y = (max_height - new_height) // 2

    background[y:y+new_height, x:x+new_width] = resized_frame

    return background
