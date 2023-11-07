import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("T.jpg")

print(f"{img=}")

img_float32 = np.float32(img)

img_dct = cv2.dct(img_float32, cv2.DCT_ROWS)
print(f"{img_dct=}")


img_dct_log = np.log(abs(img_dct))
print(f"{img_dct_log=}")

img_idct = cv2.idct(img_dct_log)
print(f"{img_idct=}")

plt.subplot(141)
plt.imshow(img, 'gray')

plt.subplot(142)
plt.imshow(img_dct, 'gray')

plt.subplot(143)
plt.imshow(img_dct_log, 'gray')

plt.subplot(144)
plt.imshow(img_idct, 'gray')
