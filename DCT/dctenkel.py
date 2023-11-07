# import required libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# read input image as grayscale
img = cv2.imread('T.jpg', 0)

# convert the grayscale to float32
imf = np.float32(img) # float conversion

# find discrete cosine transform
dct = cv2.dct(imf, cv2.DCT_ROWS)

# apply inverse discrete cosine transform
img1 = cv2.idct(dct)

# convert to uint8
img1 = np.uint8(img)

# display the image
cv2.imwrite("DCT22222.jpg", dct)
cv2.imwrite("IDCT back image2222222.jpg", img1)