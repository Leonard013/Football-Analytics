import cv2
import numpy as np
from scipy.stats import pearsonr


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
