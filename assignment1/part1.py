import logging
import config
import cv2
import numpy as np


class Part1:
    def __init__(self, capture, image_counter, video_writer):
        # capture = object to read new frame from videos
        self.capture = capture

        # counter to increment according to images written by the video_writer.
        # Used to control the length of sequence according to fps
        self.image_counter = image_counter
        self.video_writer = video_writer

    def process(self):
        """
        Process is the main function of the part1 object, that build the video part required for the first third of
        the output.
        :return:
        """
        logging.debug("inside part1.process()")
        name = self.video_writer.getBackendName()
        logging.debug("Backend Name = " + str(name))
        # print("Backend Name VW = " + str(name_vw))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        while True:
            val, frame = self.capture.read()
            if val == True:
                # cv2.imshow("Test",frame)
                if self.image_counter <= 4 * fps and \
                        (2 * self.image_counter // fps) % 2 == 1:
                    subtitle_text = "Gray image"
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # next line : for the writer to write properly - without losing the gray levels
                    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    m_frame = bgr_gray
                else:
                    subtitle_text = "Normal image"
                    m_frame = frame
                cv2.putText(m_frame, subtitle_text, (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Display the image
                self.video_writer.write(m_frame)
                # vw.write(frame)
                self.image_counter += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True
