import cv2
import config
import logging
import numpy as np

class PartTemplate:
    def __init__(self, capture, image_counter, video_writer):
        # capture = object to read new frame from videos
        self.capture = capture

        # counter to increment according to images written by the video_writer.
        # Used to control the length of sequence according to fps
        self.image_counter = image_counter
        self.video_writer = video_writer

    def write(self, image, text, pos, color=(0, 0, 0), fontScale=1, thickness=1):
        # pos = (x,y) => center of the text to be written
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # set the rectangle background to white
        rectangle_bgr = (255, 255, 255)
        label_pos = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        margin = 10
        text_width = label_pos[0][0]
        text_height = label_pos[0][1]
        # (0,0) ---> x
        # |
        # |
        # y
        text_offset_x = 0
        text_offset_y = 0
        # print(text_width)
        # print(text_height)
        _x = pos[0] - text_width//2
        _y = pos[1] - text_height//2 + margin
        box_coords = ((_x, _y), (_x + text_width + 2, _y - text_height - 2))
        cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(image, text, (_x, _y),
                    fontFace=font,
                    fontScale=fontScale,
                    color=color,
                    thickness=thickness)
        # start_point = (_x,_y)
        # end_point = (_x + label_size[0][0], _y - label_size[0][1])
        # print("x => " + str(pos[0]))
        # print("y => " + str(pos[1]))
        # image = cv2.rectangle(image, start_point, end_point, (0,255,0), thickness)


    def display_frames(self, output_shape, frames, subtitles=None):
        """
        |-------|--------|
        |  f[0] |  f[2] |
        |-------|--------|
        |  f[1] |  f[3] |
        |-------|--------|

        :param w:
        :param h:
        :param frames:
        :return:based on the shape given, arrange a view of 4 different frames by first resizing then
        concatenating them using np. Only 4 frames are currently implemented.
        """
        if len(frames) == 4:
            h = output_shape[0] // 2
            w = output_shape[1] // 2
            # logging.debug("new heigh = " + str(h))
            # logging.debug("new width = " + str(w))
            l1_resized = cv2.resize(frames[0], (w, h))
            l2_resized = cv2.resize(frames[1], (w, h))
            r1_resized = cv2.resize(frames[2], (w, h))
            r2_resized = cv2.resize(frames[3], (w, h))
            tmp_left = np.concatenate((l1_resized, l2_resized), axis=0)
            tmp_right = np.concatenate((r1_resized, r2_resized), axis=0)
            result = np.concatenate((tmp_left, tmp_right), axis=1)
            if subtitles is not None:
                pos = ((w//2, h),(w//2, 2*h), (3*w//2, h), (3*w//2, 2*h))
                self.write(result, subtitles[0], pos[0], (0, 128, 0), 0.5, 1)
                self.write(result, subtitles[1], pos[1], (0, 128, 0), 0.5, 1)
                self.write(result, subtitles[2], pos[2], (0, 128, 0), 0.5, 1)
                self.write(result, subtitles[3], pos[3], (0, 128, 0), 0.5, 1)

            return result
        else:
            raise NotImplementedError

    def is_object_present(self, frame):
        """
        :param frame: the BGR frame to analyze.
        :return: True if the object of interest (ball) is present in the frame; False otherwise
        The parameters used are selected with the tool "hough_test.py" and "hsv_capture_test.py"
        """
        dp = 1
        param1 = 255
        param2 = 8
        minDist = 600
        minRadius = 15
        maxRadius = 40
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)
        low_H = 19
        high_H = 42
        low_S = 165
        high_S = 222
        low_V = 104
        high_V = 182
        frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        r = frame_threshold

        for i in range(1, config.NUMBER_FILTER_APPLICATION):
            r = cv2.medianBlur(r, 7)

        gray = r

        circles = cv2.HoughCircles(gray,
                                   cv2.HOUGH_GRADIENT,
                                   dp=dp,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=minRadius,
                                   maxRadius=maxRadius)
        if circles is not None:
            # circles = np.uint16(np.around(circles)
            return True
            # for i in circles[0, :]:
            #     center = (i[0], i[1])
            #     # circle center
            #     cv2.circle(frame, center, 1, (0, 255, 255), 2)
            #     # circle outline
            #     radius = i[2]
            #     cv2.circle(frame, center, radius, (200, 0, 255), 2)
        return False