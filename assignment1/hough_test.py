import sys
import cv2
import numpy as np
dp = 2
param1 = 500
param2 = 24
minDist = 46
minRadius = 35
maxRadius = 48

sigma_color = 100
sigma_space = 5
video = True
image_file = "D:/Videos/part3/circle_fire.png"
small_image = cv2.imread(image_file)
template_image = small_image.copy()


def draw_fire(image, template, circle_center, radius_detected):
    radius_template = 220
    scale = radius_detected/radius_template
    new_width = int(template.shape[1] * scale)
    new_height = int(template.shape[0] * scale)
    new_size = (new_width, new_height)
    resized_template = cv2.resize(template, new_size)
    for i in range(0, resized_template.shape[0]):
        for j in range(0, resized_template.shape[1]):
            thresh = sum(resized_template[i,j])
            if thresh > 50:
                coord_width = circle_center[0] + i - new_width//2 - 3
                coord_height = circle_center[1] + j - new_height//2 + 2
                if 0 < coord_width < image.shape[0] and \
                    0 < coord_height < image.shape[1]:
                    image[coord_width, coord_height] = resized_template[i, j]
    return image

def get_circle_id(point, point_1, point_2, point_3):
    print("Points : " + str([point, point_1, point_2, point_3]))
    id = 0
    point = np.array(point, dtype=float)
    point_1 = np.array(point_1, dtype=float)
    point_2 = np.array(point_2, dtype=float)
    point_3 = np.array(point_3, dtype=float)
    dist1 = np.linalg.norm(point - point_1, ord=1)
    dist2 = np.linalg.norm(point - point_2, ord=1)
    dist3 = np.linalg.norm(point - point_3, ord=1)
    print("Distances : " + str([dist1, dist2, dist3]))
    if dist1 <= dist2 and dist1 <= dist3:
        id = 1
    elif dist2 <= dist1 and dist2 <= dist3:
        id = 2
    elif dist3 <= dist2 and dist3 <= dist2:
        id = 3
    else:
        print("Cannot decide : " + str([dist1, dist2, dist3]))
    print("ID = " + str(id))
    return id

def on_dp_trackbar(val):
    global dp
    dp = val
    dp = max(val, 1)
    cv2.setTrackbarPos("dp", "", dp)

def on_param1_trackbar(val):
    global param1
    param1 = val
    param1 = max(val, 10)
    cv2.setTrackbarPos("param1", "", param1)

def on_param2_trackbar(val):
    global param2
    param2 = val
    param2 = max(val, 10)
    cv2.setTrackbarPos("param2", "", param2)


def on_minDist_trackbar(val):
    global minDist
    minDist = val
    minDist = max(val, 1)
    cv2.setTrackbarPos("minDist", "", minDist)

def on_minRadius_trackbar(val):
    global minRadius
    minRadius = val
    minRadius = max(val, 5)
    cv2.setTrackbarPos("minDist", "", minRadius)

def on_maxRadius_trackbar(val):
    global maxRadius
    maxRadius = val
    maxRadius = max(val, 5)
    cv2.setTrackbarPos("minDist", "", maxRadius)

default_file = "D:/Videos/Test/test2.png"
video_file = "D:/Videos/part2_ball.mp4"
# video_file = "D:/Videos/part3/part3-1_downsampled_juggling.mp4"
# video_file = "D:/Videos/part3/part3-1_downsampled_MovingBalls_and_juggling.mp4"
# video_file = "D:/Videos/part3/part3-1_downsampled.mp4"

# video_file = "D:/Videos/part3/part3-2_downsampled.mp4"
# default_file = "D:/Videos/Test/im_test.png"
if video:
    filename = video_file
    cap = cv2.VideoCapture(video_file)
else:
    filename = default_file
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    backup = src.copy()
# Loads an image
# Check if image is loaded fine


window_detection_name = "cirlces detection"
cv2.namedWindow(window_detection_name)
cv2.createTrackbar("dp", window_detection_name, dp, 10, on_dp_trackbar)
cv2.createTrackbar("param1", window_detection_name, param1, 500, on_param1_trackbar)
cv2.createTrackbar("param2", window_detection_name, param2, 255, on_param2_trackbar)
cv2.createTrackbar("minDist", window_detection_name, minDist, 600, on_minDist_trackbar)
cv2.createTrackbar("minRadius", window_detection_name, minRadius, 800, on_minRadius_trackbar)
cv2.createTrackbar("maxRadius", window_detection_name, maxRadius, 800, on_maxRadius_trackbar)
counter = 0

center_positions = []
center_1_position = [(433,248)]
center_2_position = [(551,208)]
center_3_position = [(402,247)]

center_1_position = [(137, 364)]
center_2_position = [(367, 365)]
center_3_position = [(523, 368)]

# np.append(center_positions,(100,100))



while True:
    if video:
        ret, frame = cap.read()
        if frame is None:
            counter +=1
            print("Loop = " + str(counter))
            center_positions = []
            center_1_position = [(433, 248)]
            center_2_position = [(551, 208)]
            center_3_position = [(402, 247)]
            center_1_position = [(137, 364)]
            center_2_position = [(367, 365)]
            center_3_position = [(523, 368)]
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
        new_src = frame
    else:
        new_src = backup.copy()

    """
    """
    blue = new_src.copy()
    # set green and red channels to 0
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0

    green = new_src.copy()
    # set blue and red channels to 0
    green[:, :, 0] = 0
    green[:, :, 2] = 0

    red = new_src.copy()
    # set blue and green channels to 0
    red[:, :, 0] = 0
    red[:, :, 1] = 0

    yellow = green + red


    # RGB - Blue
    # cv2.imshow('B-RGB', blue)
    # cv2.imshow('G-RGB', green)
    # cv2.imshow('R-RGB', red)
    # cv2.imshow('Yellowish', yellow)

    # gray_not_filtered = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    # r = gray_not_filtered
    #TODO when object in video : this way is better

    hsv = cv2.cvtColor(new_src, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    low_H = 19
    high_H = 42
    low_S = 165
    high_S = 222
    low_V = 104
    high_V = 182

    # low_H = 21
    # high_H = 71
    # low_S = 0
    # high_S = 255
    # low_V = 0
    # high_V = 180
    #
    # low_H = 0
    # high_H = 180
    # low_S = 0
    # high_S = 255
    # low_V = 0
    # high_V = 247

    frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # r = frame_threshold
    for i in range(1, 5):
        frame_threshold = cv2.medianBlur(frame_threshold, 7)

    gray = frame_threshold.copy()
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # kernel = np.ones((5, 5), np.uint8)

    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_ellipse)
    dilatation = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_ellipse)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_ellipse)

    # new_gray = dilatation

    cv2.imshow("Dilatation ", dilatation)
    cv2.imshow("CLOSION ", closing)
    cv2.imshow("Opening", opening)
    # cv2.imshow("Modification1 ", thresholds - closing)
    # cv2.imshow("Modification2 ", thresholds - opening)
    mask = gray - dilatation
    mask2 = gray - opening

    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # new_frame = gray.copy()
    #
    # new_frame[mask2 > 0] = (0, 0, 255)
    # new_frame[mask > 0] = (0, 255, 0)

    # cv2.imshow("New_Frame", new_frame)
    new_gray = gray.copy()
    new_gray[mask2 > 0] = (0,0,0)
    new_gray[mask > 0] = (0,0,0)
    cv2.imshow("New_Gray", new_gray)

    new_gray = cv2.cvtColor(new_gray, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(new_gray,
                              cv2.HOUGH_GRADIENT,
                              dp=dp,
                              minDist= minDist,
                              param1=param1,
                              param2=param2,
                              minRadius=minRadius,
                              maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(len(circles[0,:]))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(new_src, center, 1, (0, 255, 255), 2)
            # circle outline
            radius = i[2]
            cv2.circle(new_src, center, radius, (200, 0, 255), 2)
            # center_positions.append(center)
            draw_circles = False
            circle_id = get_circle_id(center, center_1_position[-1], center_2_position[-1], center_3_position[-1])
            if draw_circles:
                new_center = (i[1], i[0])
                if circle_id == 1:
                    center_1_position=[center]
                    cv2.circle(new_src, center, radius, (0,255,0), -1)
                    draw_fire(new_src, template_image, new_center, radius)
                elif circle_id == 2:
                    center_2_position=[center]
                    cv2.circle(new_src, center, radius, (0,0,255), -1)
                    draw_fire(new_src, template_image, new_center, radius)

                elif circle_id == 3:
                    center_3_position=[center]
                    cv2.circle(new_src, center, radius, (255,0,0), -1)
                    draw_fire(new_src, template_image, new_center, radius)

                else:
                    print("center not added")

    cv2.imshow(window_detection_name, new_src)
    key = cv2.waitKey(30)

    if key == ord('q') or key == 27:
        break


