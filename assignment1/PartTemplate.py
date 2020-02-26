import cv2
import config


class PartTemplate:
    def __init__(self, capture, image_counter, video_writer):
        # capture = object to read new frame from videos
        self.capture = capture

        # counter to increment according to images written by the video_writer.
        # Used to control the length of sequence according to fps
        self.image_counter = image_counter
        self.video_writer = video_writer

    def write(self, image, text, pos, color=(255, 255, 255), fontScale=1, thickness=2):
        # pos = (x,y) => center of the text to be written
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, fontScale, thickness)
        margin = 10
        # (0,0) ---> x
        # |
        # |
        # y
        _x = pos[0] - label_size[0][0]//2
        _y = pos[1] - label_size[0][1]//2
        cv2.putText(image, text, (_x, _y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale,
                    color=color,
                    thickness=thickness)
        # start_point = (_x,_y)
        # end_point = (_x + label_size[0][0], _y - label_size[0][1])
        # print("x => " + str(pos[0]))
        # print("y => " + str(pos[1]))
        # image = cv2.rectangle(image, start_point, end_point, (0,255,0), thickness)


