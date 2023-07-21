import cv2
import numpy as np
from scipy.stats import pearsonr




class Team:
    def __init__(self, color):
        self.color = color
        self.player = []
        self.ball_possession = 0
        self.passages = 0
        self.tackles = 0
        self.outs = 0

# It returns the main color of a frame
def get_main_colors(frame):
    pixels = frame.reshape(-1, 3)
    main_color = np.mean(pixels, axis=0)
    return main_color

# It returns the color of a specific portion of the frame, used to recognize the teams' colors
def color_picker(xc,yc,wc,hc, frame_c):
    rectangular_portion = frame_c[int((yc+hc)//2.02-(0.1*(hc-yc))):int((yc+hc)//2.02+(0.1*(hc-yc))), int((xc+wc)//2-(0.1*(wc-xc))):int((xc+wc)//2+(0.1*(wc-xc)))]
    return get_main_colors(rectangular_portion)

# It returns 0 if the color is closer to the first team's color in the list, 1 otherwise
def team_recognizer(color1, color2, rec_col1):
    norm1 = np.linalg.norm(color1-rec_col1)
    norm2 = np.linalg.norm(color2-rec_col1)
    if norm1< norm2 and norm1< 140:    
        return 0
    elif norm2< norm1 and norm2< 140:
        return 1 

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

def isframe(frame):
    if frame is None:
        return False
    else:
        return True
    
def is_contour_straight(points):
    xx = [i[0] for i in points]
    yy = [i[1] for i in points]

    r, p_value = pearsonr(xx, yy)
    slope, intercept = np.polyfit(xx, yy, 1)
    return abs(r) >= 0.85, slope

# Given a list of contours, it returns the highest contour and its points.
# We use the following function to find the higthest line in the field and use it to determine if the ball is out of the field.
def find_extremes(frame, contours):
    pts = []
    highest_contour = [[0,0]]
    rightmost_contour = [[0, frame.shape[1]]]
    lowest_contour = [[frame.shape[0], frame.shape[1]]]
    leftmost_contour = [[frame.shape[0], 0]]

    highest_y = float('inf')
    rightmost_x = float('inf')
    lowest_y = float('-inf')
    leftmost_x = float('-inf')

    for contour in contours:
        # Find the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the current contour is higher than the previous highest
        pts = [[i[0][0], i[0][1]] for i in contour]
        straightness = is_contour_straight(pts)
        if y < highest_y and straightness[0] and -1 <= straightness[1] <= 0:
            highest_y = y
            highest_contour = pts
        elif x < rightmost_x and straightness[0] and 1 <= straightness[1] <= float('inf'):
            rightmost_x = x
            rightmost_contour = pts
        elif y > lowest_y and straightness[0] and 0 <= straightness[1] <= 1:
            lowest_y = y
            lowest_contour = pts
        elif x > leftmost_x and straightness[0] and -1 <= straightness[1] <= float('inf'):
            leftmost_x = x
            leftmost_contour = pts
    
    
    return highest_contour, rightmost_contour, lowest_contour, leftmost_contour


'''
We convert the input frame to HSV, we create a mask to isolate the different shades of green, in order to higlight the field's lines.
We then create a mask to isolate the white lines and we find the contours of the field's lines.
We iterate over the contours looking for the highest contour, which is the one that is closer to the top of the frame.
'''
def field_lines(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([36, 50, 50]), np.array([50, 255, 255]))
    masked = cv2.bitwise_and(im, im, mask=green_mask)
    # masked = cv2.dilate(masked, np.ones((3,3), dtype=np.uint8))

    # lwr = np.array([20, 30, 120])
    lwr = np.array([80, 100, 80])
    upp = np.array([255, 255, 255])
    masked = cv2.inRange(masked, lwr, upp)

    contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    good_cont=[]
    for i, contour in enumerate(contours):
        if cv2.contourArea(contours[i]) > 100:
            good_cont.append(contour)

    cont_pts = find_extremes(im, good_cont)
    return cont_pts[0], cont_pts[1], cont_pts[2], cont_pts[3]

