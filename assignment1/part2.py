import logging
import config
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from PartTemplate import PartTemplate


class Part2(PartTemplate):

    def process(self):
        """
        Process is the main function of the part2 object, that build the video part required for the second third of
        the output.
        :return:
        """
        logging.debug("inside part2.process()")
        name = self.video_writer.getBackendName()
        logging.debug("Backend Name = " + str(name))
        # print("Backend Name VW = " + str(name_vw))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        pos = (w // 2, h)

        while True:
            val, frame = self.capture.read()
            # logging.debug("------------------- ")
            # logging.debug("M_FRAME-SHAPE => " + str(m_frame.shape))
            # logging.debug("Val => " + str(val))
            # logging.debug("ImageCounter => " + str(self.image_counter))

            if val == True and self.image_counter <= 20 * fps:
                logging.debug("Processed part 2 = " + str(round(100 * self.image_counter / 20 / fps, 2)))
                m_frame = frame
                if self.image_counter <= 5 * fps and not config.BYPASS_SOBEL:
                    m_frame = None
                    order = None
                    ksize = None
                    scale = None
                    #todo maybe 4 images to clearly see the differences ?
                    if 0 <= self.image_counter < 1*fps :
                        frame = cv2.imread(cv2.samples.findFile(config.CALIBRATION_IMAGE), cv2.IMREAD_COLOR)
                        legend = "Visualisation of edges - example"

                        if 0 <= self.image_counter < 0.6 * fps:
                            m_frame = frame
                        elif 0.6*fps <= self.image_counter < 1 * fps:
                            m_frame = self.imprint_edges(frame, 1,3,1)
                    elif (1 <= self.image_counter < 2 * fps)  or (4 * fps <= self.image_counter < 5 * fps):
                        order = 1
                        ksize = 3
                        scale = 1
                    elif 2* fps <= self.image_counter < 3 * fps:
                        order = 1
                        ksize = 7
                        scale = 1
                    elif 3* fps <= self.image_counter < 4 * fps:
                        order = 3
                        ksize = 5
                        scale = 1
                    else:
                        order = 1
                        ksize = 3
                        scale = 5

                    # m_frame is updated only if needed.
                    if order is not None and\
                            ksize is not None and \
                            scale is not None:
                        m_frame = self.imprint_edges(frame, order, ksize, scale)
                        legend = "O(dx,dy)=" + str(order) + ", k=" + str(ksize) + ", scale=" + str(scale)
                    self.write(m_frame, legend, (w//2, h))
                    # cv2.imshow("frame", m_frame)

                elif 5*fps < self.image_counter <= 15 * fps and not config.BYPASS_HOUGH:
                    logging.debug("Hough")

                    # case 1:
                    """
                    parameters are chosen to isolate limited amount of circles
                    decreasing param2 decreases the threshold for the accumulator, that defines circles detected. It 
                     globally increases the amount of detection.
                     minDist parameter specifies a minimal distance between detected circles. decreasing allows 
                     closer circle, such as this situation. THis is something we usually want to avoid.
                     param1 is the parameter used in the edge detection (canny). decreasing it makes more circles 
                     detected.
                     
                     increasing d makes at first more circles detected, as the 
                     It offers a new scale for the other parameters, as it is the inverse of the ratio of the 
                     accumulator resolution to the image resolution.
                    """
                    gray_not_filtered = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    r = gray_not_filtered
                    legend = "default"
                    if 5*fps <= self.image_counter < 7*fps:
                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR,
                                                    config.SIGMA_SPACE)
                        dp = 1
                        param1 = 160
                        param2 = 30
                        minDist = 50
                        minRadius = 46
                        maxRadius = 50
                        legend = "centered to leaf - 1;160;30;50;10;100"
                        #########
                        gray = r
                        # cv2.imshow("grayFiltered", gray)
                        edges = cv2.Canny(gray, param1, param1 / 2)
                        # cv2.imshow("canny", edges)
                        circles_detected = cv2.HoughCircles(gray,
                                                            cv2.HOUGH_GRADIENT,
                                                            dp=dp,
                                                            minDist=minDist,
                                                            param1=param1,
                                                            param2=param2,
                                                            minRadius=minRadius,
                                                            maxRadius=maxRadius)
                        if circles_detected is not None:
                            circles_detected = np.uint16(np.around(circles_detected))
                            for i in circles_detected[0, :]:
                                print(i)
                                center = (i[0], i[1])
                                # circle center
                                cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                                # circle outline
                                radius = i[2]
                                cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                        # cv2.imshow("Hough", m_frame)
                        self.write(m_frame, legend, (w//2, h), thickness=2)
                        #########
                    elif 7*fps <= self.image_counter < 8*fps:
                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR,
                                                    config.SIGMA_SPACE)

                        dp = 1
                        param1 = 160
                        param2 = 30
                        minDist = 1
                        minRadius = 30
                        maxRadius = 50
                        legend = "Min distance between detection very low"
                        #########
                        gray = r
                        # cv2.imshow("grayFiltered", gray)
                        edges = cv2.Canny(gray, param1, param1 / 2)
                        # cv2.imshow("canny", edges)
                        circles_detected = cv2.HoughCircles(gray,
                                                            cv2.HOUGH_GRADIENT,
                                                            dp=dp,
                                                            minDist=minDist,
                                                            param1=param1,
                                                            param2=param2,
                                                            minRadius=minRadius,
                                                            maxRadius=maxRadius)
                        if circles_detected is not None:
                            circles_detected = np.uint16(np.around(circles_detected))
                            for i in circles_detected[0, :]:
                                print(i)
                                center = (i[0], i[1])
                                # circle center
                                cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                                # circle outline
                                radius = i[2]
                                cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                        # cv2.imshow("Hough", m_frame)
                        self.write(m_frame, legend, (w//2, h), thickness=2)
                        #########
                    elif 8*fps <= self.image_counter < 9*fps:
                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR,
                                                    config.SIGMA_SPACE)
                        dp = 1
                        param1 = 160
                        param2 = 15
                        minDist = 50
                        minRadius = 10
                        maxRadius = 100
                        legend = "More (noisy) detected (accumulator threshold decreased)"
                        #########
                        gray = r
                        # cv2.imshow("grayFiltered", gray)
                        edges = cv2.Canny(gray, param1, param1 / 2)
                        # cv2.imshow("canny", edges)
                        circles_detected = cv2.HoughCircles(gray,
                                                            cv2.HOUGH_GRADIENT,
                                                            dp=dp,
                                                            minDist=minDist,
                                                            param1=param1,
                                                            param2=param2,
                                                            minRadius=minRadius,
                                                            maxRadius=maxRadius)
                        if circles_detected is not None:
                            circles_detected = np.uint16(np.around(circles_detected))
                            for i in circles_detected[0, :]:
                                print(i)
                                center = (i[0], i[1])
                                # circle center
                                cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                                # circle outline
                                radius = i[2]
                                cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                        # cv2.imshow("Hough", m_frame)
                        self.write(m_frame, legend, (w//2, h), thickness=2)
                        #########
                    elif 9*fps <= self.image_counter < 11*fps:
                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR,
                                                    config.SIGMA_SPACE)
                        dp = 1
                        param1 = 90
                        param2 = 30
                        minDist = 50
                        minRadius = 10
                        maxRadius = 100
                        legend = "Threshold for internal edge detection decreased -> more detection"

                        #########
                        gray = r
                        # cv2.imshow("grayFiltered", gray)
                        edges = cv2.Canny(gray, param1, param1 / 2)
                        # cv2.imshow("canny", edges)
                        circles_detected = cv2.HoughCircles(gray,
                                                            cv2.HOUGH_GRADIENT,
                                                            dp=dp,
                                                            minDist=minDist,
                                                            param1=param1,
                                                            param2=param2,
                                                            minRadius=minRadius,
                                                            maxRadius=maxRadius)
                        if circles_detected is not None:
                            circles_detected = np.uint16(np.around(circles_detected))
                            for i in circles_detected[0, :]:
                                print(i)
                                center = (i[0], i[1])
                                # circle center
                                cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                                # circle outline
                                radius = i[2]
                                cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                        # cv2.imshow("Hough", m_frame)
                        self.write(m_frame, legend, (w // 2, h), thickness=2)
                        #########
                    #
                    # elif 11*fps <= self.image_counter < 12*fps:
                    #     for i in range(1, config.NUMBER_FILTER_APPLICATION):
                    #         r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR,
                    #                                 config.SIGMA_SPACE)
                    #     dp = 2
                    #     param1 = 201
                    #     param2 = 40
                    #     minDist = 160
                    #     minRadius = 229
                    #     maxRadius = 251
                    #     legend = "min and max Radius both high => only large circles"
                    #     #########
                    #     gray = r
                    #     cv2.imshow("grayFiltered", gray)
                    #     edges = cv2.Canny(gray, param1, param1 / 2)
                    #     cv2.imshow("canny", edges)
                    #     circles_detected = cv2.HoughCircles(gray,
                    #                                         cv2.HOUGH_GRADIENT,
                    #                                         dp=dp,
                    #                                         minDist=minDist,
                    #                                         param1=param1,
                    #                                         param2=param2,
                    #                                         minRadius=minRadius,
                    #                                         maxRadius=maxRadius)
                    #     if circles_detected is not None:
                    #         circles_detected = np.uint16(np.around(circles_detected))
                    #         for i in circles_detected[0, :]:
                    #             print(i)
                    #             center = (i[0], i[1])
                    #             # circle center
                    #             cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                    #             # circle outline
                    #             radius = i[2]
                    #             cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                    #     cv2.imshow("Hough", m_frame)
                    #     self.write(m_frame, legend, (w // 2, h), thickness=2)
                    #     #########
                    # # # elif 12*fps <= self.image_counter < 13*fps:
                    # elif 12*fps <= self.image_counter < 15*fps:
                    #     legend = "including sharpening"
                    #     dp = 2
                    #     param1 = 220
                    #     param2 = 181
                    #     minDist = 20
                    #     minRadius = 50
                    #     maxRadius = 95
                    #     for i in range(1, config.NUMBER_FILTER_APPLICATION):
                    #         r = cv2.bilateralFilter(r,  config.BILATERAL_FILTER_D, config.SIGMA_COLOR, config.SIGMA_SPACE)
                    #         kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    #         r = cv2.filter2D(r, -1, kernel)
                    #
                    #     #########
                    #     gray = r
                    #     cv2.imshow("grayFiltered", gray)
                    #     edges = cv2.Canny(gray, param1, param1 / 2)
                    #     cv2.imshow("canny", edges)
                    #     circles_detected = cv2.HoughCircles(gray,
                    #                                         cv2.HOUGH_GRADIENT,
                    #                                         dp=dp,
                    #                                         minDist=minDist,
                    #                                         param1=param1,
                    #                                         param2=param2,
                    #                                         minRadius=minRadius,
                    #                                         maxRadius=maxRadius)
                    #     if circles_detected is not None:
                    #         circles_detected = np.uint16(np.around(circles_detected))
                    #         for i in circles_detected[0, :]:
                    #             print(i)
                    #             center = (i[0], i[1])
                    #             # circle center
                    #             cv2.circle(m_frame, center, 1, (200, 0, 255), 1)
                    #             # circle outline
                    #             radius = i[2]
                    #             cv2.circle(m_frame, center, radius, (200, 0, 255), 2)
                    #     cv2.imshow("Hough", m_frame)
                    #     self.write(m_frame, legend, (w//2, h), thickness=2)
                    #     #########
                    elif 11 * fps <= self.image_counter <= 15 * fps:
                        r = gray_not_filtered.copy()
                        r_sharp = gray_not_filtered.copy()

                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r = cv2.bilateralFilter(r,
                                                    config.BILATERAL_FILTER_D,
                                                    config.SIGMA_COLOR,
                                                    config.SIGMA_SPACE)
                        for i in range(1, config.NUMBER_FILTER_APPLICATION):
                            r_sharp = cv2.bilateralFilter(r_sharp,
                                                          config.BILATERAL_FILTER_D,
                                                          config.SIGMA_COLOR,
                                                          config.SIGMA_SPACE)
                            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                            r_sharp = cv2.filter2D(r_sharp, -1, kernel)

                        gray = r
                        hloc = frame.shape[0] // 2
                        wloc = frame.shape[1] // 2

                        legend_l1 = "Left 1"
                        legend_l2 = "Left 2"
                        legend_r1 = "Right 1"
                        legend_r2 = "Right 2"

                        # cv2.imshow("grayFiltered", gray)
                        # edges = cv2.Canny(gray, param1, param1 / 2)
                        # cv2.imshow("canny", edges)

                        # detecting circles with different parameters set for the 4 images
                        # L1) normal only leaf
                        l1_prep = frame.copy()
                        gray_l1 = gray.copy()
                        param_l1 = [1, 160, 30, 50, 40, 50]
                        # L2) two leaves ?
                        l2_prep = frame.copy()
                        param_l2 = [2, 350, 24, 46, 57, 62]
                        gray_l2 = gray.copy()
                        # R1) only orchidea
                        r1_prep = frame.copy()
                        gray_r1 = r_sharp.copy()
                        param_r1 = [2, 220, 181, 20, 47, 95]
                        # todo: filtering =
                        #  r = cv2.bilateralFilter(r, 5, 100, 5) x3
                        #  + sharpening
                        # R2) large circles
                        r2_prep = frame.copy()
                        gray_r2 = gray.copy()
                        param_r2 = [2, 295, 30, 600, 253, 259]
                        # todo: filtering =
                        #  r = cv2.bilateralFilter(r, 5, 100, 5)

                        params = [param_l1, param_l2, param_r1, param_r2]
                        grays = [gray_l1, gray_l2, gray_r1, gray_r2]
                        output_frames=[l1_prep, l2_prep, r1_prep, r2_prep]
                        for index in range(0,4):
                            # logging.debug("i = " + str(index) + " -> params = " + str(params[index]))
                            circles_detected = cv2.HoughCircles(grays[index],
                                                                cv2.HOUGH_GRADIENT,
                                                                dp=params[index][0],
                                                                minDist=params[index][3],
                                                                param1=params[index][1],
                                                                param2=params[index][2],
                                                                minRadius=params[index][4],
                                                                maxRadius=params[index][5])
                            if circles_detected is not None:
                                circles_detected = np.uint16(np.around(circles_detected))
                                for i in circles_detected[0, :]:
                                    center = (i[0], i[1])
                                    # circle center
                                    cv2.circle(output_frames[index], center, 1, (200, 0, 255), 1)
                                    # circle outline
                                    radius = i[2]
                                    cv2.circle(output_frames[index], center, radius, (200, 0, 255), 2)

                        # cv2.imshow("output_frames_0",output_frames[0])
                        # cv2.imshow("l1_prep",l1_prep)
                        # cv2.imshow("l2_prep",l2_prep)
                        # cv2.imshow("r1_prep",r1_prep)
                        # cv2.imshow("r2_prep",r2_prep)

                        l1 = cv2.resize(output_frames[0], (wloc, hloc))
                        l2 = cv2.resize(output_frames[1], (wloc, hloc))
                        r1 = cv2.resize(output_frames[2], (wloc, hloc))
                        r2 = cv2.resize(output_frames[3], (wloc, hloc))
                        m_frame = self.display_frames(frame.shape,
                                                      (l1, l2, r1, r2),
                                                      (legend_l1, legend_l2, legend_r1, legend_r2))
                        # cv2.imshow("Hough", m_frame)
                        # self.write(m_frame, legend, (w // 2, h), thickness=2)
                elif 15*fps < self.image_counter<=20*fps:
                    if 15*fps < self.image_counter <= 17*fps:
                        # show object
                        if self.is_object_present(frame):

                            frame_matching = frame.copy()
                            info = np.iinfo(frame_matching.dtype)  # Get the information of the incoming image type
                            frame_matching = frame_matching.astype(np.float64) / info.max  # normalize the data to 0 - 1
                            frame_matching = 255 * frame_matching  # Now scale by 255
                            img = frame_matching.astype(np.uint8)
                            m_frame=frame
                            template = cv2.imread(config.TEMPLATE_FILE, 0)
                            width, height = template.shape[::-1]
                            result_matching = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                                template,
                                                                cv2.TM_CCORR_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_matching)

                            rect_corner_top_left = max_loc
                            rect_corner_bottom_right = (rect_corner_top_left[0] + width, rect_corner_top_left[1] + height)
                            cv2.rectangle(m_frame, rect_corner_top_left, rect_corner_bottom_right, (0, 255, 0), 2, 8, 0)
                        else:
                            m_frame = frame

                    elif 17*fps < self.image_counter <= 20*fps:
                        # show object and max likelihood
                        frame_matching = frame.copy()
                        info = np.iinfo(frame_matching.dtype)  # Get the information of the incoming image type
                        frame_matching = frame_matching.astype(np.float64) / info.max  # normalize the data to 0 - 1
                        frame_matching = 255 * frame_matching  # Now scale by 255
                        img = frame_matching.astype(np.uint8)
                        m_frame = frame
                        template = cv2.imread(config.TEMPLATE_FILE, 0)
                        width, height = template.shape[::-1]
                        result_matching = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                            template,
                                                            cv2.TM_CCOEFF_NORMED).astype(np.float64)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_matching)

                        rect_corner_top_left = max_loc
                        rect_corner_bottom_right = (rect_corner_top_left[0] + width, rect_corner_top_left[1] + height)
                        m_frame_tmp = cv2.copyMakeBorder(result_matching,
                                                        45, 45, 45, 45,
                                                        cv2.BORDER_CONSTANT, value=(0,0,0))
                        m_frame_tmp3 = np.dstack((m_frame_tmp, m_frame_tmp, m_frame_tmp))



                        m_frame= np.array(((m_frame_tmp3 - m_frame_tmp3.min()) / (m_frame_tmp3.max() -
                                                                                 m_frame_tmp3.min()) * 255 + 0)).astype(np.uint8)
                        cv2.rectangle(m_frame, rect_corner_top_left, rect_corner_bottom_right, (0, 255, 0), 2, 8, 0)

                        # cv2.imshow("M_FRAME", m_frame)


                else:
                    subtitle_text = "Else case - Normal Image"
                    m_frame = frame
                    self.write(m_frame, subtitle_text, (w // 2, h // 2))
                # print(m_frame.shape)
                # logging.debug("M_FRAME_SHAPE = " + str(m_frame.shape))
                # logging.debug("M_FRAME_DTYPE" + str(m_frame.dtype))
                self.video_writer.write(m_frame)
                self.image_counter += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True




