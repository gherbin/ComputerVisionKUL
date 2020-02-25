import logging
import config
import part1
import part2
import part3
import cv2

def main():
    logging.basicConfig(level=logging.DEBUG)

    input_file = config.INPUT_FILE
    output_file = config.OUTPUT_FILE

    capture = cv2.VideoCapture(input_file)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    logging.debug("FPS = " + str(fps))
    logging.debug("h = " + str(h))
    logging.debug("w = " + str(w))
    image_counter = 0

    fourcc = cv2.VideoWriter_fourcc(*config.FOURCC)
    output_video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    logging.debug("Is Opened [True]? " + str(output_video_writer.isOpened()))

    basic = part1.Part1(capture, image_counter, output_video_writer)
    basic.process()
    # part2()
    # part3()

    logging.info("ImageCounter = " + str(basic.image_counter))
    logging.debug("Is Opened ? " + str(output_video_writer.isOpened()))

    output_video_writer.release()
    capture.release()
    cv2.destroyAllWindows()

main()
