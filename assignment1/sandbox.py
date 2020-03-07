import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

default_file = "D:/Videos/Test/test2.png"
# default_file = "D:/Videos/Test/im_test.png"
# default_file = "D:/Videos/Test/calibration.png"

filename = default_file
window_detection_name = "test"

# Loads an image
src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
# src = np.float32(src)

print("SRC TYPE = " + str(src.dtype))
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3,3),0)


# for param1 in [50]:
#     for param2 in [10, 20, 30, 40, 50]:
cv2.namedWindow(window_detection_name)
order = 1
ksize = 3
scale = 1
# while True:
# backup = src.copy()
# frame = backup.copy()

# frame = src
m_frame = src
cv2.imshow(window_detection_name, src)
src = cv2.GaussianBlur(src, (5, 5), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


dx = cv2.Sobel(gray,
               ddepth=cv2.CV_64F,
               dx=1 * order,
               dy=0 * order,
               ksize=ksize,
               scale=scale,
               delta=0)
dy = cv2.Sobel(gray,
               ddepth=cv2.CV_64F,
               dx=0 * order,
               dy=1 * order,
               ksize=ksize,
               scale=scale,
               delta=0)
mag, orn = cv2.cartToPolar(dx, dy, angleInDegrees=True)
cv2.imshow(" magnitude ", mag)
cv2.imshow(" orientation" , orn)
# "Orn, 255, Mag"


orn_normalized = np.array(((orn - orn.min())/(orn.max() - orn.min()) * 180 + 0)).astype(np.uint8)
mag_normalized = np.array(((mag - mag.min())/(mag.max() - mag.min()) * 255 + 0)).astype(np.uint8)
mag_normalized[mag_normalized<40] = 0

mat0 = orn_normalized
mat1 = (np.ones((480,852))*255).astype(np.uint8)
mat2 = mag_normalized

HSV = np.dstack((mat0, mat1, mat2))
result = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
cv2.imshow("trial", result)

print("============ MAGNITUDE============ ")
print("Shape = " + str(mag.shape))
print("Type = " + str(mag.dtype))
print("min = " + str(mag.min()))
print("max = " + str(mag.max()))
print("sum = " + str(mag.sum()))
print("mean = " + str(mag.mean()))
# print("============ NORMALIZED MAGNITUDE============ ")
# print("Shape = " + str(mag_normalized.shape))
# print("Type = " + str(mag_normalized.dtype))
# print("min = " + str(mag_normalized.min()))
# print("max = " + str(mag_normalized.max()))
# print("sum = " + str(mag_normalized.sum()))
# print("mean = " + str(mag_normalized.mean()))
# print("============ NORMALIZED MAGNITUDE 2 ============ ")
# print("Shape = " + str(mag_normalized2.shape))
# print("Type = " + str(mag_normalized2.dtype))
# print("min = " + str(mag_normalized2.min()))
# print("max = " + str(mag_normalized2.max()))
# print("sum = " + str(mag_normalized2.sum()))
# print("mean = " + str(mag_normalized2.mean()))
#

print("============ ORIENTATION============ ")
print("Shape = " + str(orn.shape))
print("Type = " + str(orn.dtype))
print("min = " + str(orn.min()))
print("max = " + str(orn.max()))
print("sum = " + str(orn.sum()))
print("mean = " + str(orn.mean()))
# print("============ NORMALIZED ORIENTATION 1 ============ ")
# print("Shape = " + str(orn_normalized.shape))
# print("Type = " + str(orn_normalized.dtype))
# print("min = " + str(orn_normalized.min()))
# print("max = " + str(orn_normalized.max()))
# print("sum = " + str(orn_normalized.sum()))
# print("mean = " + str(orn_normalized.mean()))
# print("============ NORMALIZED ORIENTATION 2 ============ ")
#
# print("Shape = " + str(orn_normalized2.shape))
# print("Type = " + str(orn_normalized2.dtype))
# print("min = " + str(orn_normalized2.min()))
# print("max = " + str(orn_normalized2.max()))
# print("sum = " + str(orn_normalized2.sum()))
# print("mean = " + str(orn_normalized2.mean()))
#


print("============ HSV ============ ")

print("Shape = " + str(HSV.shape))
print("Type = " + str(HSV.dtype))
print(HSV)
print(HSV[1::])
print("min = " + str(HSV[1::].min()))
print("min = " + str(HSV[2::].min()))
print("min = " + str(HSV[3::].min()))

print("max = " + str(HSV[1::].max()))
print("max = " + str(HSV[2::].max()))
print("max = " + str(HSV[3::].max()))
print("sum = " + str(HSV.sum()))
print("mean = " + str(HSV.mean()))

i1 = "namei1"
i2 = "namei2"
i3 = "namei3"
plt.matshow(mag,i1)
# plt.matshow(mag_normalized,i2)
plt.matshow(cv2.convertScaleAbs(mag),i3)

j1 = "namej1"
j2 = "namej2"
j3 = "namej3"

plt.matshow(orn,j1)
# plt.matshow(orn_normalized,j2)
plt.matshow(cv2.convertScaleAbs(orn), j3)

plt.show()

key = cv2.waitKey(0)


# if key == ord('q') or key == 27:
#     break