import logging
import config
import cv2
import numpy as np
import os

from PartTemplate import PartTemplate


class Part1(PartTemplate):
    def add_noise(self, frame, noise_type, mean=0, sigma=50, amplitude=0.5):
        if noise_type == "gauss":
            # start by creating an array of 0 values
            gaussian_noise = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.randn(gaussian_noise, mean, sigma)
            gaussian_noise = (gaussian_noise * amplitude).astype(np.uint8)
            # logging.debug("Frame shape = " + str(frame.shape))
            # logging.debug("Gaussian_noise shape = " + str(gaussian_noise.shape))
            bgr_gaussian_noise=cv2.cvtColor(gaussian_noise, cv2.COLOR_GRAY2BGR)
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
        pos = (w // 2, h)


        while True:
            val, frame = self.capture.read()
            m_frame = frame
            if val == True and self.image_counter <= 20 * fps:
                # cv2.imshow("Test",frame)
                logging.debug("Processed part 1 = " +str(round(100*self.image_counter / 20/fps,2)))

                if self.image_counter <= 4 * fps:
                    if (self.image_counter // (fps/2)) % 2 == 1:
                        subtitle_text = "Gray image"
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # next line : for the writer to write properly - without losing the gray levels
                        bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        m_frame = bgr_gray
                    else:
                        m_frame = frame
                elif 4*fps < self.image_counter <= 12 * fps:
                    # 4 -> 12 : Filter experimentations
                    # 4 -> 7  : Gaussian
                    # 7 -> 10 : Bilateral
                    # 11 & 12 : comparison between strongest gaussian, strongest bilateral

                    mean = 0
                    sigma = 75
                    amplitude = 0.5
                    bgr_noisy = self.add_noise(frame, "gauss", mean=mean, sigma=sigma, amplitude = amplitude)

                    if 4*fps < self.image_counter <= 6 * fps:
                        #  Gaussian filters
                        ks = ((3,27),(7,7),(17,17))
                        sigmaX = (1.5, 1.5, 3)
                        sigmaY = (4.5, 1.5, 3)

                        hloc = frame.shape[0] // 2
                        wloc = frame.shape[1] // 2
                        l2_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r1_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r2_prep = cv2.resize(bgr_noisy, (wloc, hloc))

                        l2 = self.apply_gaussian(l2_prep, ks[0], sigmaX=sigmaX[0], sigmaY=sigmaY[0])
                        # legend_l2 = "Gaussian Filter: k=" + str(ks[0])  # + ", sig" + str(sigmaX)
                        legend_l2 = "Asymmetric (larger in Y dir)"

                        r1 = self.apply_gaussian(r1_prep, ks[1], sigmaX=sigmaX[1], sigmaY=sigmaY[1])
                        # legend_r1 = "Gaussian Filter: k=" + str(ks[1])  # + ", sig" + str(sigmaX)
                        legend_r1 = "symmetric, medium gaussian (k = " + str(ks[1]) + ")"

                        r2 = self.apply_gaussian(r2_prep, ks[2], sigmaX=sigmaX[2], sigmaY=sigmaY[2])
                        # legend_r2 = "Gaussian Filter: k=" + str(ks[2])  # + ", sig" + str(sigmaX)
                        legend_r2 = "symmetric, large gaussian (k = " + str(ks[2]) + ")"

                        legend_frame = "Original"
                        legend_noisy = "Noisy original"

                        # new_height = h // 2
                        # new_width = w // 2
                        m_frame = self.display_frames(frame.shape,
                                                      (bgr_noisy, l2, r1, r2),
                                                      (legend_noisy, legend_l2 , legend_r1, legend_r2))

                    elif 6*fps < self.image_counter <= 8*fps and not config.BYPASS_BILATERAL:
                        hloc = frame.shape[0] // 2
                        wloc = frame.shape[1] // 2
                        l2_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r1_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r2_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        d = -1  # computed from sigmaSpace

                        sigmaColor = 5
                        sigmaSpace = 15
                        l2 = l2_prep
                        for i in range(1, config.NUMBER_BILATERAL_FILTER_APPLICATION):
                            l2 = cv2.bilateralFilter(l2, d, sigmaColor, sigmaSpace)
                        legend_l2 = "Bilateral F., sigma_space (" + \
                                    str(sigmaSpace) + ") > sigma_color (" + str(sigmaColor) + ")"
                        sigmaColor = 8
                        sigmaSpace = 8
                        r1 = r1_prep
                        for i in range(1, config.NUMBER_BILATERAL_FILTER_APPLICATION):
                            r1 = cv2.bilateralFilter(r1, d, sigmaColor, sigmaSpace)
                        legend_r1 = "Bilateral F., sigma_space (" + \
                                    str(sigmaSpace) + ") == sig_color (" + str(sigmaColor) + ")"

                        sigmaColor = 50
                        sigmaSpace = 5
                        r2 = r2_prep
                        for i in range(1, config.NUMBER_BILATERAL_FILTER_APPLICATION):
                            r2 = cv2.bilateralFilter(r2, d, sigmaColor, sigmaSpace)
                        legend_r2 = "Bilateral F., sigma_color (" + \
                                    str(sigmaColor) + ") >> sigma_space (" + str(sigmaSpace)+")"

                        m_frame = self.display_frames(frame.shape,
                                                      (bgr_noisy, l2, r1, r2),
                                                      ("Noisy original", legend_l2, legend_r1, legend_r2))

                    elif 8*fps < self.image_counter <= 12*fps and not config.BYPASS_BILATERAL:
                        # m_frame = cv2.bilateralFilter(bgr_noisy, d=-1, sigmaColor=5, sigmaSpace=30)
                        ks = ((7, 7), (17, 17))
                        sigmaX = 0

                        hloc = frame.shape[0] // 2
                        wloc = frame.shape[1] // 2
                        l1_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        l2_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r1_prep = cv2.resize(bgr_noisy, (wloc, hloc))
                        r2_prep = cv2.resize(bgr_noisy, (wloc, hloc))

                        l1 = self.apply_gaussian(l1_prep, ks[0], sigmaX)
                        legend_l1 = "Gaussian : k=" + str(ks[0])  # + ", sig" + str(sigmaX)

                        l2 = self.apply_gaussian(l2_prep, ks[1], sigmaX)
                        legend_l2 = "Gaussian : k=" + str(ks[1])  # + ", sig" + str(sigmaX)

                        d = -1  # computed from sigmaSpace
                        sigmaColor = 8
                        sigmaSpace = 8
                        r1 = r1_prep
                        for i in range(1, config.NUMBER_BILATERAL_FILTER_APPLICATION):
                            r1 = cv2.bilateralFilter(r1, d, sigmaColor, sigmaSpace)
                        legend_r1 = "Bilateral filter (x" + str(config.NUMBER_BILATERAL_FILTER_APPLICATION) + \
                                    "), both sigmas = " + str(sigmaColor)

                        sigmaColor = 50
                        sigmaSpace = 5
                        r2 = r2_prep
                        for i in range(1, config.NUMBER_BILATERAL_FILTER_APPLICATION):
                            r2 = cv2.bilateralFilter(r2, d, sigmaColor, sigmaSpace)
                        legend_r2 = "Bilateral filter (x" + str(config.NUMBER_BILATERAL_FILTER_APPLICATION) + \
                                    "), sigma color = " + str(sigmaColor) + \
                                    ", sigma space = " + str(sigmaSpace)

                        m_frame = self.display_frames(frame.shape,
                                                      (l1, l2, r1, r2),
                                                      (legend_l1, legend_l2, legend_r1, legend_r2))
                        self.write(m_frame, "blurry", (w//3, h//2 + 50), (0,0,255))
                        self.write(m_frame, "cartoon-like", (3*w//4, h//2 + 50), (0,0,255))
                        # cv2.imshow("Test 10-12", m_frame)
                    else:
                        subtitle_text = "Bypass - Normal Image"
                        m_frame = frame
                        self.write(m_frame, subtitle_text, (w // 2, h // 2))
                elif 12 * fps < self.image_counter <= 20 * fps:
                    # 1) 1s     RGB
                    # 2) 2s     HSV
                    # 3) 2s     binary frames
                    #
                    if 12 * fps < self.image_counter <= 13 * fps:
                        m_frame = frame
                    elif 13*fps < self.image_counter <= 14.5 * fps:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        m_frame = hsv
                    elif 14.5*fps < self.image_counter <= 17 * fps:
                        # binary frame in HSV
                        new_frame = self.apply_gaussian(frame, (7,7), 10)

                        if config.THRESHOLD_COLOR_SPACE is "HSV":
                            hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
                            low_H = 0
                            low_S = 146
                            low_V = 62
                            high_H = 180
                            high_S = 255
                            high_V = 255
                            thresholds = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
                            m_frame = cv2.cvtColor(thresholds, cv2.COLOR_GRAY2BGR)
                            legend = "Binary frame from HSV color space thresholding"
                            self.write(m_frame, legend, (w//2, h), fontScale=1)
                            # cv2.imshow(" test binary", m_frame)
                        elif config.THRESHOLD_COLOR_SPACE is "GRAY":
                            sigmaColor = 30
                            sigmaSpace = 3
                            bilateral = frame  # gaussian
                            for i in range(1, 5):
                                bilateral = cv2.bilateralFilter(bilateral, -1, sigmaColor, sigmaSpace)
                            useable = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
                            ret2, thresh_low = cv2.threshold(useable, config.THRESHOLD_LOW, 255, cv2.THRESH_BINARY)
                            ret3, thresh_high = cv2.threshold(useable, config.THRESHOLD_HIGH, 255, cv2.THRESH_BINARY)
                            result = thresh_low - thresh_high
                            bgr_thresh_low = cv2.cvtColor(thresh_low, cv2.COLOR_GRAY2BGR)
                            bgr_thresh_high = cv2.cvtColor(thresh_high, cv2.COLOR_GRAY2BGR)
                            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

                            m_frame = self.display_frames(frame.shape,
                                                          (frame, bgr_thresh_low, result_bgr, bgr_thresh_high),
                                                          ("Original", "Low Threshold", "High - Low", "High Threshold"))
                        else:
                            raise NotImplementedError("Wrong color space in config")
                    elif 17 * fps < self.image_counter <= 20 * fps:
                        # in HSV, grab by the color
                        # use of morphological operations in order to improve grabbing
                        yellow = (30, 255, 255)

                        low_H = 0
                        low_S = 146
                        low_V = 62
                        high_H = 100
                        high_S = 255
                        high_V = 255

                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        thresholds = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
                        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        closing = cv2.morphologyEx(thresholds, cv2.MORPH_CLOSE, kernel_ellipse)
                        # cv2.imshow("test binary2", closing)
                        dilatation = cv2.morphologyEx(thresholds, cv2.MORPH_DILATE, kernel_ellipse)
                        opening = cv2.morphologyEx(thresholds, cv2.MORPH_OPEN, kernel_ellipse)

                        # cv2.imshow("Dilatation ", dilatation)
                        # cv2.imshow("CLOSION ", closing)
                        # cv2.imshow("Modification1 ", thresholds - closing)
                        # cv2.imshow("Modification2 ", thresholds - opening)
                        mask = thresholds - dilatation
                        mask2 = thresholds - opening
                        mask3 = thresholds - closing
                        thresholds = cv2.cvtColor(thresholds, cv2.COLOR_GRAY2BGR)
                        new_frame = thresholds

                        new_frame[mask > 0] = (0, 255, 0)
                        new_frame[mask2 > 0] = (0, 0, 255)
                        new_frame[mask3 > 0] = (255, 0, 0)

                        # cv2.imshow("Modification3 ", new_frame)
                        m_frame = new_frame

                    else:
                        self.write(m_frame, "TO BE MODIFIED", (w // 2, h // 2), (0, 0, 0), 3, 3)

                else:
                    subtitle_text = "Bypass - Normal Image"
                    m_frame = frame
                    self.write(m_frame, subtitle_text, (w//2, h//2))

                # self.write(m_frame, subtitle_text, (w//2, h))
                #
                # Display the image
                # logging.debug("M_FRAME SHAPE = " + str(m_frame.shape))
                self.video_writer.write(m_frame)
                self.image_counter += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True
