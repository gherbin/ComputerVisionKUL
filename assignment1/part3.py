import logging
import config
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from PartTemplate import PartTemplate


class Part3(PartTemplate):
    def __init__(self, captures, image_counter, video_writer):
        super(Part3, self).__init__(captures[0], image_counter, video_writer)
        self.center_1_position = [(137, 364)]
        self.center_2_position = [(367, 365)]
        self.center_3_position = [(523, 368)]
        self.captures = captures

    def set_capture(self, number):
        self.capture = self.captures[number]

    def process(self):
        """
        Process is the main function of the part2 object, that build the video part required for the second third of
        the output.
        :return:
        """
        logging.debug("inside part3.process()")
        # name = self.video_writer.getBackendName()
        # logging.debug("Backend Name = " + str(name))
        # print("Backend Name VW = " + str(name_vw))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        pos = (w // 2, h)

        while True:
            val, frame = self.capture.read()
            # logging.debug("------------------- ")
            # logging.debug("M_FRAME-SHAPE => " + str(m_frame.shape))
            # logging.debug("Val => " + str(val))
            # logging.debug("ImageCounter => " + str(self.image_counter))

            if val == True and self.image_counter <= config.TIME_LIMIT_END * fps:
                logging.debug("Processed part 3 = " + str(round(100 * self.image_counter / 20 / fps, 2)))

                m_frame = frame
                # if 0 <= self.image_counter < 1 * fps:
                #     m_frame = frame
                if 0 <= self.image_counter < config.PART3_SWITCH_SCENE_0 * fps and \
                        not config.BYPASS_JUGGLE:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    # cv2.imshow("HSV", hsv)
                    low_H = 19
                    high_H = 42
                    low_S = 165
                    high_S = 222
                    low_V = 104
                    high_V = 182

                    dp = 2
                    param1 = 500
                    param2 = 24
                    minDist = 46
                    minRadius = 11
                    maxRadius = 40

                    image_file = config.TEMPLATE_FILE_CIRCLE_FIRE
                    template_image = cv2.imread(image_file)

                    frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
                    r = frame_threshold

                    for i in range(1, 5):
                        r = cv2.medianBlur(r, 11)

                    gray = r
                    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    kernel = np.ones((5, 5), np.uint8)

                    my_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))
                    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, my_kernel)
                    new_gray = closing

                    circles = cv2.HoughCircles(new_gray,
                                               cv2.HOUGH_GRADIENT,
                                               dp=dp,
                                               minDist=minDist,
                                               param1=param1,
                                               param2=param2,
                                               minRadius=minRadius,
                                               maxRadius=maxRadius)

                    if circles is not None and len(circles[0, :]) > 1:
                        circles = np.uint16(np.around(circles))
                        # print(len(circles[0, :]))
                        for i in circles[0, :]:
                            center = (i[0], i[1])
                            # circle center
                            cv2.circle(m_frame, center, 1, (0, 255, 255), 2)
                            # circle outline
                            radius = i[2]
                            # cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                            # center_positions.append(center)
                            draw_circles = True
                            circle_id = self.get_circle_id(center,
                                                           self.center_1_position[-1],
                                                           self.center_2_position[-1],
                                                           self.center_3_position[-1])
                            if draw_circles:
                                new_center = (i[1], i[0])
                                if circle_id == 1:
                                    self.center_1_position = [center]
                                    cv2.circle(m_frame, center, radius, (0,255,0), -1)
                                    self.draw_fire(m_frame, template_image, new_center, radius)
                                elif circle_id == 2:
                                    self.center_2_position = [center]
                                    cv2.circle(m_frame, center, radius, (0,0,255), -1)
                                    self.draw_fire(m_frame, template_image, new_center, radius)

                                elif circle_id == 3:
                                    self.center_3_position = [center]
                                    cv2.circle(m_frame, center, radius, (255,0,0), -1)
                                    self.draw_fire(m_frame, template_image, new_center, radius)

                                else:
                                    logging.debug("No circle added")
                elif config.PART3_SWITCH_SCENE_0*fps <= self.image_counter <= config.TIME_LIMIT_END*fps and not config.BYPASS_HOCKEY:
                    self.set_capture(1)
                    hsv_ball = [0, 180, 0, 255, 0, 247]
                    hsv_line = [0, 72, 27, 255, 42, 255]

                    hough_ball = [2, 500, 20, 600, 22, 40]
                    hough_line = [1, np.pi / 180, 140, None, 0, 0]

                    flag_goal = False
                    f_frame = cv2.medianBlur(frame,7)
                    hsv = cv2.cvtColor(f_frame, cv2.COLOR_BGR2HSV)

                    circles = self.get_circles(hsv, hsv_ball, hough_ball)

                    p3 = None
                    min_radius = 0
                    if circles is not None and len(circles[0, :] == 1):
                        flag_goal = True
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            center = (i[0], i[1])
                            # circle center
                            cv2.circle(m_frame, center, 1, (0, 255, 255), 2)
                            # circle outline
                            radius = i[2]
                            cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                            p3 = np.asarray(center)
                            min_radius = radius

                    #########################################
                    lines = self.get_lines(hsv, hsv_line, hough_line)
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
                            cv2.line(m_frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                            p1 = np.asarray(pt1)
                            p2 = np.asarray(pt2)
                            if p3 is not None:
                                d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                                va = p1 - p3
                                vb = p2 - p3
                                if np.cross(va, vb) > 0 and d > min_radius:
                                    flag_goal = flag_goal and True
                                else:
                                    flag_goal = flag_goal and False

                    if flag_goal:
                        m_frame[50:150, 100:200] = (0, 255, 0)
                        self.write(m_frame, "GOAL", (150,100), (0,0,0),0.5,3, draw_background=False)
                    else:
                        m_frame[50:150, 100:200] = (0, 0, 255)
                        self.write(m_frame, "NO GOAL", (150,100), (0,0,0),0.5,3, draw_background=False)

                # logging.debug("M_FRAME_SHAPE = " + str(m_frame.shape))
                # logging.debug("M_FRAME_DTYPE" + str(m_frame.dtype))
                # cv2.imshow("Part3_m_frame",m_frame)
                self.video_writer.write(m_frame)
                self.image_counter += 1

            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True

    def get_circle_id(self, point, point_1, point_2, point_3):
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

    def draw_fire(self, image, template, circle_center, radius_detected):
        radius_template = config.RADIUS_TEMPLATE
        scale = radius_detected / radius_template
        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)
        new_size = (new_width, new_height)
        resized_template = cv2.resize(template, new_size)
        for i in range(0, resized_template.shape[0]):
            for j in range(0, resized_template.shape[1]):
                thresh = sum(resized_template[i, j])
                if thresh > 50:
                    coord_x = circle_center[0] + i - new_width // 2 - 3
                    coord_y = circle_center[1] + j - new_height // 2 + 2
                    if 0 < coord_x < image.shape[0] and \
                            0 < coord_y < image.shape[1]:
                        image[coord_x, coord_y] = resized_template[i, j]
        return image

    def get_lines(self, hsv, hsv_line, hough_line):
        frame_threshold_line = cv2.inRange(hsv,
                                           (hsv_line[0], hsv_line[2], hsv_line[4]),
                                           (hsv_line[1], hsv_line[3], hsv_line[5]))
        r_line = frame_threshold_line
        for i in range(1, 5):
            r_line = cv2.medianBlur(r_line, 7)

        gray_line = r_line.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
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

    def get_circles(self, hsv, hsv_ball, hough_ball):
        # cv2.imshow("HSV", hsv)
        frame_threshold_ball = cv2.inRange(hsv,
                                           (hsv_ball[0], hsv_ball[2], hsv_ball[4]),
                                           (hsv_ball[1], hsv_ball[3], hsv_ball[5]))
        r_ball = frame_threshold_ball
        for i in range(1, 5):
            r_ball = cv2.medianBlur(r_ball, 5)

        gray_ball = r_ball.copy()
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel = np.ones((5, 5), np.uint8)
        # my_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))
        # closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, my_kernel)
        closing = cv2.morphologyEx(gray_ball, cv2.MORPH_CLOSE, kernel_ellipse)
        # dilatation = cv2.morphologyEx(gray_ball, cv2.MORPH_DILATE, kernel_ellipse)
        # opening = cv2.morphologyEx(gray_ball, cv2.MORPH_OPEN, kernel_ellipse)

        # cv2.imshow("Dilatation ", dilatation)
        # cv2.imshow("CLOSION ", closing)
        # cv2.imshow("Modification1 ", thresholds - closing)
        # cv2.imshow("Modification2 ", thresholds - opening)
        # mask = gray_ball - dilatation
        # mask2 = gray_ball - opening
        #
        # gray_ball[mask2 > 0] = 0
        # gray_ball[mask > 0] = 0
        # cv2.imshow("New_Gray", gray_ball)
        # cv2.imshow("DIFF", gray_ball - r_ball)

        gray_ball = closing

        circles = cv2.HoughCircles(gray_ball,
                                   cv2.HOUGH_GRADIENT,
                                   dp=hough_ball[0],
                                   minDist=hough_ball[3],
                                   param1=hough_ball[1],
                                   param2=hough_ball[2],
                                   minRadius=hough_ball[4],
                                   maxRadius=hough_ball[5])
        return circles

