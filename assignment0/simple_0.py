# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:07:17 2020

@author: Geoffroy Herbin
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

plt.close('all')

v1 = np.zeros((100,100))
v2 = v1 + 100
v3 = v1 + 200

v= np.concatenate((v1,v2,v3), 1)
fig1 = plt.figure()
plt.imshow(v, cmap="gray")

fig2 = plt.figure()
ax1 = fig2.add_subplot(131)  # Top left side
ax2 = fig2.add_subplot(132)  # Top middle side
ax3 = fig2.add_subplot(133)  # Top right side
ax4 = fig2.add_subplot(231)
ax5 = fig2.add_subplot(232)
ax6 = fig2.add_subplot(233)

axesTop = [ax1, ax2, ax3]
axesBot = [ax4, ax5, ax6]

noise = [10, 50, 100]
v_noisy = [0,0,0]
for i in range(0, len(axesTop)):
    v_noisy[i] = v + noise[i] * (np.random.rand(100,300) - 0.5)
    axesTop[i].imshow(v_noisy[i], cmap = "gray")


for j in range(0, len(axesBot)):
    axesBot[j].imshow(gaussian_filter(v_noisy[j], sigma=5), cmap = "gray")

print("+==========================")
image = cv2.imread("images.png")
cv2.imshow("image_name", image)

plt.show()
