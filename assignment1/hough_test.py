import sys
import cv2
import numpy as np
dp = 2
param1 = 255
param2 = 24
minDist = 46
minRadius = 57
maxRadius = 62

sigma_color = 100
sigma_space = 5
video = True

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

"HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None):"
"""
    HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    .   @brief Finds circles in a grayscale image using the Hough transform.
    .   
    .   The function finds circles in a grayscale image using a modification of the Hough transform.
    .   
    .   Example: :
    .   @include snippets/imgproc_HoughLinesCircles.cpp
    .   
    .   @note Usually the function detects the centers of circles well. However, it may fail to find correct
    .   radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
    .   you know it. Or, you may set maxRadius to a negative number to return centers only without radius
    .   search, and find the correct radius using an additional procedure.
    .   
    .   @param image 8-bit, single-channel, grayscale input image.
    .   @param circles Output vector of found circles. Each vector is encoded as  3 or 4 element
    .   floating-point vector \f$(x, y, radius)\f$ or \f$(x, y, radius, votes)\f$ .
    .   @param method Detection method, see #HoughModes. Currently, the only implemented method is #HOUGH_GRADIENT
    .   @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
    .   dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
    .   half as big width and height.
    .   @param minDist Minimum distance between the centers of the detected circles. If the parameter is
    .   too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
    .   too large, some circles may be missed.
    .   @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT , it is the higher
    .   threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    .   @param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT , it is the
    .   accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
    .   false circles may be detected. Circles, corresponding to the larger accumulator values, will be
    .   returned first.
    .   @param minRadius Minimum circle radius.
    .   @param maxRadius Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns
    .   centers without finding the radius.
    .   
    .   @sa fitEllipse, minEnclosingCircle
    """
default_file = "D:/Videos/Test/test2.png"
video_file = "D:/Videos/part2_ball.mp4"
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
    cv2.imshow('B-RGB', blue)
    cv2.imshow('G-RGB', green)
    cv2.imshow('R-RGB', red)
    cv2.imshow('Yellowish', yellow)

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
    frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    r = frame_threshold

    for i in range(1, 5):
        # r = cv2.GaussianBlur(r, (7, 7), 0)
        # r = cv2.GaussianBlur(r, (25,25),0)
        r = cv2.medianBlur(r, 7)
        # r = cv2.bilateralFilter(r,5, 100, 5)
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # r = cv2.filter2D(r, -1, kernel)

    gray = r
    cv2.imshow("grayFiltered", gray)
    edges = cv2.Canny(r, param1, param1/2)

    cv2.imshow("canny", edges)

    circles = cv2.HoughCircles(gray,
                              cv2.HOUGH_GRADIENT,
                              dp=dp,
                              minDist= minDist,
                              param1=param1,
                              param2=param2,
                              minRadius=minRadius,
                              maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(new_src, center, 1, (0, 255, 255), 2)
            # circle outline
            radius = i[2]
            cv2.circle(new_src, center, radius, (200, 0, 255), 2)

    cv2.imshow("gray", gray)
    cv2.imshow(window_detection_name, new_src)

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break