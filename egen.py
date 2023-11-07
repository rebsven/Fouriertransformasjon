import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist





#MAIN
image_path = r"test.jpg"
image = cv2.imread(image_path, 0)

# calculating the discrete Fourier transform
DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# reposition the zero-frequency component to the spectrum's middle
shift = np.fft.fftshift(DFT)
row, col = image.shape
center_row, center_col = row // 2, col // 2
 
# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(shift[:,:,0],shift[:,:,1]))
 
