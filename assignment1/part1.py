import logging
import config
import cv2
import numpy as np
import os

from PartTemplate import PartTemplate


class Part1(PartTemplate):
    def add_noise(self, frame, noise_type, mean=128, sigma=50, amplitude=0.5):
        if noise_type == "gauss":
            # start by creating an array of 0 values
            gaussian_noise = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.randn(gaussian_noise, mean, sigma)
            gaussian_noise = (gaussian_noise * amplitude).astype(np.uint8)
            # logging.debug("Frame shape = " + str(frame.shape))
            # logging.debug("Gaussian_noise shape = " + str(gaussian_noise.shape))
            bgr_gaussian_noise=cv2.cvtColor(gaussian_noise,cv2.COLOR_GRAY2BGR)
            return cv2.add(frame, bgr_gaussian_noise)
        elif noise_type == "salt_pepper":
            raise NotImplementedError
        elif noise_type == "random_uniform":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def apply_gaussian(self,frame, ksize=(5,5), sigmaX=10, sigmaY=0):
        gaussian = cv2.GaussianBlur(src=frame, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        return gaussian

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
                        (self.image_counter // fps) % 2 == 1:
                    subtitle_text = "Gray image"
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # next line : for the writer to write properly - without losing the gray levels
                    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    m_frame = bgr_gray
                elif 4*fps < self.image_counter:# <= 12 * fps:
                    subtitle_text = "Working on it"
                    # convert to gray - 3 layers
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    # convert to noisy gray - 3 layers
                    bgr_noisy_gray = self.add_noise(frame, "gauss", mean=128, sigma=20)

                    # noisy = self.add_noise(frame, "gauss")

                    if self.image_counter < 6*fps:
                        ksize = (1, 1)
                        sigmaX = 20
                        sigmaY = 0
                    # elif self.image_counter < 6*fps:
                    #     ksize = (7, 7)
                    #     sigmaX = 20
                    #     sigmaY = 0
                    elif self.image_counter < 7.5*fps:
                        ksize = (7, 7)
                        sigmaX = 20
                        sigmaY = 0
                    elif self.image_counter < 10*fps:
                        ksize=(15,15)
                        sigmaX = 20
                    elif self.image_counter < 12*fps:
                        ksize=(15,15)
                        sigmaX = 10
                    elif self.image_counter > 12 * fps:
                        ksize = (15, 15)
                        sigmaX = 1
                    subtitle_text = "Gaussian Filter : k="+ str(ksize) + "sigma=" + str(sigmaX)

                    gaussian_blur = self.apply_gaussian(bgr_noisy_gray, ksize, sigmaX)
                    # print(cv2.getGaussianKernel(ksize=ksize[0], sigma=sigmaX))
                    # x[1:3, 0:2]
                    new_height = h // 10
                    new_width = w // 10
                    # frame_resized = cv2.resize(bgr_noisy_gray, (new_height, new_width))

                    left_half =bgr_noisy_gray[0:480, 0:852//4, :]
                    right_half = gaussian_blur[0:480, 852//4:852,:]

                    # logging.debug("Left Half size " + str(left_half.shape))
                    # logging.debug("Right Half size " + str(right_half.shape))

                    # m_frame = bgr_noisy_gray
                    m_frame = np.concatenate((left_half, right_half), axis=1)
                    self.write(m_frame, "TO BE MODIFIED", (w//2, h//2), (0,0,255), 3, 3)


                else:
                    subtitle_text = "Normal image"
                    m_frame = frame

                # cv2.putText(m_frame, subtitle_text, (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.write(m_frame, subtitle_text, (w//2, h))
                # Display the image
                self.video_writer.write(m_frame)
                # vw.write(frame)
                self.image_counter += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True
