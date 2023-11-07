#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('T.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # center

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# Tester under her
mellomshifted = np.fft.ifftshift(mask)
mellomting = np.fft.ifftshift(dft)
# mellomting = cv2.idft(mellomshifted)
mellom = cv2.magnitude(mellomting[:,:,0],mellomting[:,:,1])

# mask
mellomshifted = np.fft.ifftshift(dft)
# mellomting = cv2.idft(mellomshifted)
t2 = np.fft.ifftshift(mask)[:, :, 1]

# mask
t3 = np.fft.ifftshift(mask)[:, :, 0]
# t3 = mask
# t3 = cv2.idft(t2)
# t4 = cv2.magnitude(t3[:,:,0],t3[:,:,1])
# Tester ferdig

plt.subplot(151)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(152)
plt.imshow(mellom, cmap = 'gray')
plt.title('Frekvensbilde'), plt.xticks([]), plt.yticks([])

plt.subplot(153)
plt.imshow(t2, cmap = 'gray')
plt.title('Frekvensbilde'), plt.xticks([]), plt.yticks([])

plt.subplot(154)
plt.imshow(t3, cmap = 'gray')
plt.title('Frekvensbilde'), plt.xticks([]), plt.yticks([])

plt.subplot(155)
plt.imshow(img_back, cmap = 'gray')
plt.title('Komprimert'), plt.xticks([]), plt.yticks([])

plt.savefig('DFT2.jpg')       