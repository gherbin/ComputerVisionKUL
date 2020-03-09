import sys
import cv2
import numpy as np
dp = 2
param1 = 471
param2 = 21
minDist = 46
minRadius = 35
maxRadius = 48

rho = 1
theta = 1
threshold = 200
srn = None
stn = None
min_theta = np.pi/4
max_theta = 3*np.pi/4



sigma_color = 100
sigma_space = 5
video = True
image_file = "D:/Videos/part3/circle_fire.png"
small_image = cv2.imread(image_file)
template_image = small_image.copy()
hsv_ball = [0,180,0,255,0,247]
hsv_line = [0,72,27,255,42,255]

hough_ball = [2,500,20,600,22,40]
hough_line = [1, np.pi/180, 140, None, 0,0]


default_file = "D:/Videos/Test/test2.png"
video_file = "D:/Videos/part2_ball.mp4"
video_file = "D:/Videos/part3/part3-1_downsampled_juggling.mp4"
video_file = "D:/Videos/part3/part3-1_downsampled_MovingBalls_and_juggling.mp4"
video_file = "D:/Videos/part3/part3-2_downsampled.mp4"
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


window_detection_name = "lines detection"
cv2.namedWindow(window_detection_name)


counter = 0

# np.append(center_positions,(100,100))

def get_lines(hsv, hsv_line, hough_line):
    hsv = cv2.medianBlur(hsv, 11)
    frame_threshold_line = cv2.inRange(hsv,
                                       (hsv_line[0], hsv_line[2], hsv_line[4]),
                                       (hsv_line[1], hsv_line[3], hsv_line[5]))
    r_line = frame_threshold_line
    for i in range(1, 5):
        r_line = cv2.medianBlur(r_line, 7)

    gray_line = r_line.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    closing = cv2.morphologyEx(gray_line, cv2.MORPH_CLOSE, kernel)
    gray_line = np.bitwise_not(closing)
    cv2.imshow("New_Gray", gray_line)

    edges = cv2.Canny(gray_line, 100, 200, None, 3)

    lines = cv2.HoughLines(edges,
                           hough_line[0],
                           hough_line[1],
                           hough_line[2],
                           hough_line[3],
                           hough_line[4],
                           hough_line[5])
    return lines


def get_circles(hsv, hsv_ball, hough_ball):
    # cv2.imshow("HSV", hsv)
    hsv = cv2.medianBlur(hsv,7)
    frame_threshold_ball = cv2.inRange(hsv,
                                       (hsv_ball[0], hsv_ball[2], hsv_ball[4]),
                                       (hsv_ball[1], hsv_ball[3], hsv_ball[5]))
    r_ball = frame_threshold_ball
    for i in range(1, 5):
        r_ball = cv2.medianBlur(r_ball, 5)

    gray_ball = r_ball.copy()
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(gray_ball, cv2.MORPH_CLOSE, kernel_ellipse)
    dilatation = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_ellipse)
    opening = cv2.morphologyEx(gray_ball, cv2.MORPH_OPEN, kernel_ellipse)

    # cv2.imshow("Dilatation ", dilatation)
    # cv2.imshow("CLOSION ", closing)
    # cv2.imshow("Modification1 ", thresholds - closing)
    # cv2.imshow("Modification2 ", thresholds - opening)
    mask = gray_ball - dilatation
    mask2 = gray_ball - opening

    gray_ball[mask2 > 0] = 0
    gray_ball[mask > 0] = 0
    cv2.imshow("New_Gray", gray_ball)
    cv2.imshow("DIFF", gray_ball - r_ball)
    # gray_ball = dilatation
    circles = cv2.HoughCircles(gray_ball,
                               cv2.HOUGH_GRADIENT,
                               dp=hough_ball[0],
                               minDist=hough_ball[3],
                               param1=hough_ball[1],
                               param2=hough_ball[2],
                               minRadius=hough_ball[4],
                               maxRadius=hough_ball[5])
    return circles


while True:
    if video:
        ret, frame = cap.read()
        if frame is None:
            counter +=1
            print("Loop = " + str(counter))
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
        new_src = frame
    else:
        new_src = backup.copy()
    flag_goal = False
    hsv = cv2.cvtColor(new_src, cv2.COLOR_BGR2HSV)

    circles = get_circles(hsv, hsv_ball, hough_ball)


    p3 = None
    min_radius = 0
    if circles is not None and len(circles[0,:] == 1):
        flag_goal = True
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(new_src, center, 1, (0, 255, 255), 2)
            # circle outline
            radius = i[2]
            cv2.circle(new_src, center, radius, (200, 0, 255), 2)
            # center_positions.append(center)
            p3 = np.asarray(center)
            min_radius = radius

    #########################################
    lines = get_lines(hsv, hsv_line, hough_line)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.math.cos(theta)
            b = np.math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(new_src, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            p1 = np.asarray(pt1)
            p2 = np.asarray(pt2)
            if p3 is not None:
                d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                va = p1-p3
                vb = p2-p3
                if np.cross(va, vb) > 0 and d > min_radius:
                    flag_goal = flag_goal and True
                else:
                    flag_goal = flag_goal and False

    if flag_goal:
        new_src[50:150, 100:200] = (0,255,0)
    else:
        new_src[50:150,100:200] = (0,0,255)

    cv2.imshow(window_detection_name, new_src)
    key = cv2.waitKey(1)



    if key == ord('q') or key == 27:
        break


