import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
print("Hello World !")


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        _shape = image.shape
        mean = 0
        # var = 0.1
        # sigma = var**0.5
        gauss = np.random.normal(mean, 3, _shape)
        gauss = gauss.reshape(_shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    # elif noise_typ == "poisson":
    #     vals = len(np.unique(image))
    #     vals = 2 ** np.ceil(np.log2(vals))
    #     noisy = np.random.poisson(image * vals) / float(vals)
    #     return noisy
    # elif noise_typ =="speckle":
    #     row,col,ch = image.shape
    #     gauss = np.random.randn(row,col,ch)
    #     gauss = gauss.reshape(row,col,ch)
    #     noisy = image + image * gauss
    #     return noisy

video_file = "D:/Videos/downsampled.mp4"

cap = cv2.VideoCapture(video_file)

fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
minutes = int(duration/60)
seconds = round(duration % 60)
length = f'{minutes}:{seconds}'

print("FPS = " + str(fps))
print("h = " + str(h))
print("w = " + str(w))
print("frame = " + str(frame_count))
print("duration = " + str(duration))


# print(cap.getBackendName())
# [val, frame] = cap.read()

# cv2.imshow("MyVideo", frame)
# cv2.waitKey(1) & 0xFF == ord('q')
# time.sleep(5)

# performance is really important => keep it efficient
# keep the frame rate correct

t1 = time.time()
image_counter = 0

while True:
    val, frame = cap.read()
    if frame is None:
        break
    # TODO ask if we can use imutils library

    # frame = imutils.resize(frame, width=640)
    frame = imutils.rotate_bound(frame, 90)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_noisy = noisy("gauss", gray).astype('uint8')

    if image_counter > 100 and image_counter < 300:
        cv2.circle(frame, (int(h//2), int(w//2)), 50, (0,0,255), 5)
    # if image_counter > 200:
    #     param = int(image_counter // 100)
    #     kernel = np.ones((param, param), np.float32) / param*param
    #     dst = cv2.filter2D(frame, -1, kernel)
    #     cv2.imshow("DST", dst)

    blur = cv2.blur(frame_noisy,(10,10))
    # cv2.imshow("BLURRED", blur)
    gaussian = cv2.GaussianBlur(frame_noisy,(9,9),0)
    # gaussian = cv2.GaussianBlur(frame_noisy, ksize=(11,11), sigmaX=10, sigmaY=10)
    bilateral = cv2.bilateralFilter(frame_noisy, 9, 75, 75)
    # cv2.imshow("Bilateral", bilateral)

    cv2.putText(frame_noisy, 'NOISY', (10, 500), cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 255), 2)
    cv2.putText(blur, 'BLUR', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(gaussian, 'GAUSSIAN', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(bilateral, 'BILATERAL', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    filtered = np.concatenate((frame_noisy, blur, gaussian, bilateral), axis=1)

    cv2.imshow("Filtered", filtered)


    ###########################################################################
    img = gray
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    gaussianblur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(gaussianblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot all the images and their histograms
    # images = [img, 0, th1,
    #           img, 0, th2,
    #           blur, 0, th3]
    cv2.imshow("Thresholds ", np.concatenate((th1, th2, th3), axis=1))
    # titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
    #           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
    #           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    # for i in range(3):
    #     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    # plt.show()


    # cv2.circle(frame, (100, 100), 50, (255, 0, 0), 3)

    # waitKey will wait for a key to be pressed for at least 1 ms, and only the last 8 digits are kept
    # this is a way to make the framerate as the original
    #TODO is there another way ?
    # 10 instead of 1 seems to do the trick
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    res_frame = np.concatenate((frame,hsv), axis=1)



    # cv2.imshow("MyVideo", frame)
    # cv2.imshow("HSV", res_frame)
    # cv2.imshow("GrayScale", gray)


    image_counter+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
t2 = time.time()
print("Elapsed_time = " + str(round((t2-t1), 2)) + " [s]")
cap.release()
cv2.destroyAllWindows()