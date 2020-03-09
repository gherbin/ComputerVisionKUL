import cv2


def main():
    video_file_1 = "D:/Videos/part2-HD_1.mp4"
    video_file_2 = "D:/Videos/part2-HD_2.mp4"
    # video_file_3 = "D:/Videos/part3/hockey/part3-2_d.mp4"
    # video_file_4 = "D:/Videos/part3/hockey/part3-2_z.mp4"

    output_file = "D:/Videos/part2-HD.mp4"
    cap1 = cv2.VideoCapture(video_file_1)
    cap2 = cv2.VideoCapture(video_file_2)
    # cap3 = cv2.VideoCapture(video_file_3)
    # cap4 = cv2.VideoCapture(video_file_4)
    caps = (cap1, cap2) # , cap3, cap4)

    fps = round(float(cap1.get(cv2.CAP_PROP_FPS)), 2)
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (w,h))

    for cap in caps:
        while True:
            val, frame = cap.read()
            if val == True:
                out.write(frame)
            else:
                print("break")
                break
        cap.release()

    out.release()
    cv2.destroyAllWindows()


main()
