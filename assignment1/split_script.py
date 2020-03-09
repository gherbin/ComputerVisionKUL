import cv2


def main():
    video_file = "D:/Videos/part3/part3-1_downsampled_MovingBalls_and_juggling.mp4"
    output_file = "D:/Videos/part3/part3-1_downsampled.mp4"
    cap = cv2.VideoCapture(video_file)
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (w,h))

    counter = 0
    counter1 = 0
    counter2 = 0
    while True:
        val, frame = cap.read()
        if val == True:
            if counter <= 1 * fps:
                # out.write(frame)
                print("nothing")
            elif 1 * fps < counter <= 3 * fps:
                out.write(frame)
            elif 3*fps < counter <= 8.5*fps:
                out.write(frame)
            elif 8.5*fps < counter <= 11.3*fps:
                print("nothing")
            elif 11.3*fps < counter <= 14.5*fps:
                counter2+=1
                out.write(frame)
            elif 14.5*fps < counter :
                print("nothing")
            else:
                out.write(frame)
            counter += 1
        else:
            break
    print(counter2)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


main()
