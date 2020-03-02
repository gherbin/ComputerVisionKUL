import cv2


class Downsampler:

    def downsample(self, video_file, output_file, ratio):

        cap = cv2.VideoCapture(video_file)
        fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_height = int(h // ratio)
        new_width = int(w // ratio)

        out = cv2.VideoWriter(output_file, fourcc, fps, (new_width,new_height))
        print("converting h : " + str(h) + " -> " + str(new_height))
        print("converting w : " + str(w) + " -> " + str(new_width))

        while True:
            val, frame = cap.read()
            if val == True:
                frame_resized = cv2.resize(frame, (new_width, new_height))
                out.write(frame_resized)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()