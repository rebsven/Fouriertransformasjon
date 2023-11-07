import numpy as np
import cv2
from matplotlib import pyplot as plt
 
# read the input image
# you can specify the path to image
image_path = r"test.jpg"
image = cv2.imread(image_path, 0)
 
# calculating the discrete Fourier transform
DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
 
# reposition the zero-frequency component to the spectrum's middle
shift = np.fft.fftshift(DFT)
row, col = image.shape
center_row, center_col = row // 2, col // 2
 
# create a mask with a centered square of 1s
mask = np.zeros((row, col, 2), np.uint8)
mask[center_row - 100:center_row + 50, center_col - 60:center_col + 60] = 1
 
# put the mask and inverse DFT in place.
fft_shift = shift * mask
fft_ifft_shift = np.fft.ifftshift(fft_shift)
imageThen = cv2.idft(fft_ifft_shift)
 
# calculate the magnitude of the inverse DFT
imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])


 
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Compute the discrete Fourier Transform of the image
#fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
 
# Shift the zero-frequency component to the center of the spectrum
#fourier_shift = np.fft.fftshift(fourier)
 
# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(shift[:,:,0],shift[:,:,1]))
 
# Scale the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
 
# Display the magnitude of the Fourier Transform
cv2.imwrite('Fourier Transform.jpg', magnitude)









# visualize the original image and the magnitude spectrum
plt.figure(figsize=(10,10))
#plt.subplot(121), plt.imshow(image, cmap='gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(121), plt.imshow(magnitude, cmap='gray')
plt.title('Fourier transformert'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imageThen, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig("hei.jpg")