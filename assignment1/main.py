import logging
import config
import part1
import part2
import part3
import cv2

def main():
    logging.basicConfig(level=logging.DEBUG)

    output_file = config.OUTPUT_FILE

    capture_part1 = cv2.VideoCapture(config.INPUT_FILE_PART1)
    capture_part2 = cv2.VideoCapture(config.INPUT_FILE_PART2)
    capture_part31 = cv2.VideoCapture(config.INPUT_FILE_PART3_1)
    capture_part32 = cv2.VideoCapture(config.INPUT_FILE_PART3_2)

    fps = float(capture_part1.get(cv2.CAP_PROP_FPS))
    h = int(capture_part1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(capture_part1.get(cv2.CAP_PROP_FRAME_WIDTH))

    #TODO assert that all videos capture have the same fps, h, w

    logging.debug("FPS = " + str(fps))
    logging.debug("h = " + str(h))
    logging.debug("w = " + str(w))
    image_counter = 0

    fourcc = cv2.VideoWriter_fourcc(*config.FOURCC)
    output_video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    logging.debug("Is Opened [True]? " + str(output_video_writer.isOpened()))
    #
    # basic = part1.Part1(capture_part1, image_counter, output_video_writer)
    # basic.process()
    #
    # detection = part2.Part2(capture_part2, image_counter, output_video_writer)
    # detection.process()

    fun = part3.Part3((capture_part31,capture_part32), image_counter, output_video_writer)
    fun.process()

    # logging.info("ImageCounter = " + str(detection.image_counter))
    # logging.debug("Is Opened ? " + str(output_video_writer.isOpened()))

    output_video_writer.release()
    capture_part1.release()
    capture_part2.release()
    cv2.destroyAllWindows()

main()
