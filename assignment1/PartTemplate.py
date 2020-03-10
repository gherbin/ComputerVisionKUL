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

    def write(self, image, text, pos, color=(255, 255, 255), fontScale=1, thickness=1, draw_background=True):
        """
        Encapsulation of the cv2.puText function, including shaping and proper organization of text.
        :param image: image to write text into
        :param text: text to write in image
        :param pos: position of the center of the text
        :param color: color of the text
        :param fontScale: see cv2.putText
        :param thickness: see cv2.putText
        :param draw_background: option to draw (or not) a background, cosmetic
        :return: nothing
        """
        if text is None:
            return

        # pos = (x,y) => center of the text to be written
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_pos = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        margin_y = 3
        text_width = label_pos[0][0]
        text_height = label_pos[0][1]
        # (0,0) ---> x
        # |
        # |
        # y

        _x = pos[0] - text_width//2
        _y = pos[1] - text_height//2
        # set the rectangle background to white
        if draw_background:
            rectangle_bgr = (0, 0, 0)
            box_coords = ((_x, _y + margin_y), (_x + text_width + 2, _y - text_height - 2))
            cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

        cv2.putText(image, text, (_x, _y),
                    fontFace=font,
                    fontScale=fontScale,
                    color=color,
                    thickness=thickness)


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
            logging.debug("new height = " + str(h))
            logging.debug("new width = " + str(w))
            l1_resized = cv2.resize(frames[0], (w, h))
            l2_resized = cv2.resize(frames[1], (w, h))
            r1_resized = cv2.resize(frames[2], (w, h))
            r2_resized = cv2.resize(frames[3], (w, h))
            tmp_left = np.concatenate((l1_resized, l2_resized), axis=0)
            tmp_right = np.concatenate((r1_resized, r2_resized), axis=0)
            result = np.concatenate((tmp_left, tmp_right), axis=1)
            if subtitles is not None:
                pos = ((w//2, h),(w//2, 2*h), (3*w//2, h), (3*w//2, 2*h))
                self.write(result, subtitles[0], pos[0], (255, 255, 255), 0.4, 1)
                self.write(result, subtitles[1], pos[1], (255, 255, 255), 0.4, 1)
                self.write(result, subtitles[2], pos[2], (255, 255, 255), 0.4, 1)
                self.write(result, subtitles[3], pos[3], (255, 255, 255), 0.4, 1)

            return result
        else:
            raise NotImplementedError

    def imprint_edges(self, frame, order, ksize, scale):
        """
        used in Sobel.
        :param frame: frame on which to imprint the edges
        :param order: parameter of sobel filter
        :param ksize: "
        :param scale: "
        :return: m_frame, the frame with color edges imprinted
        """
        src = cv2.GaussianBlur(frame, (3,3), 0)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        dx = cv2.Sobel(gray,
                       ddepth=cv2.CV_64F,
                       dx=1 * order,
                       dy=0 * order,
                       ksize=ksize,
                       scale=scale,
                       delta=0)
        dy = cv2.Sobel(gray,
                       ddepth=cv2.CV_64F,
                       dx=0 * order,
                       dy=1 * order,
                       ksize=ksize,
                       scale=scale,
                       delta=0)

        mag, orn = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        # thresh param determined visually
        thresh = 50
        # normalization to have more beautiful colors
        # offset defined experimentally
        orn_normalized = np.array(((orn - orn.min()) / (orn.max() - orn.min()) * 180 + 30)).astype(np.uint8)
        mag_normalized = np.array(((mag - mag.min()) / (mag.max() - mag.min()) * 255 + 0)).astype(np.uint8)
        mag_normalized[mag_normalized < thresh] = 0

        mat0 = orn_normalized
        mat1 = (np.ones(orn.shape) * 255).astype(np.uint8)
        mat2 = mag_normalized
        hsv_edges = np.dstack((mat0, mat1, mat2))
        m_frame = cv2.addWeighted(cv2.cvtColor(hsv_edges, cv2.COLOR_HSV2BGR), 1,
                                  cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.3,
                                  gamma = 0)
        # cv2.imshow("sobel", m_frame)
        return m_frame

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
            return True
        return False
