import cv2


class Downsampler:

    def downsample(self, video_file, output_file, ratio):

        cap = cv2.VideoCapture(video_file)
        fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_height = h // ratio
        new_width = w // ratio
        out = cv2.VideoWriter(output_file, fourcc, fps, (new_height,new_width))
        print("converting h : " + str(h) + " -> " + str(new_height))
        print("converting w : " + str(w) + " -> " + str(new_width))

        while True:
            val, frame = cap.read()
            if val == True:
                frame_resized = cv2.resize(frame, (new_height, new_width))
                out.write(frame_resized)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()