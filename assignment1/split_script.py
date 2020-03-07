import cv2


def main():
    video_file = "D:/Videos/part2_downsampled_long.mp4"
    output_file = "D:/Videos/part2_ball.mp4"
    cap = cv2.VideoCapture(video_file)
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (w,h))

    counter = 0
    while True:
        val, frame = cap.read()
        if val == True:
            counter += 1
            if counter <= 15*fps:
                # out.write(frame)
                continue
            elif 15*fps<=counter <= 30*fps:
                continue
            elif 30*fps < counter <= 36*fps:
                out.write(frame)
            else:
                out.write(frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


main()
