import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
# %matplotlib inline

src_path = 'test.jpg'
dst_path = 'test.jpg'
names = 'test.jpg'

fshift_values = []
dct_values = []
image_values = []

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
for idx, name in enumerate(names[1:-1]):
    i = 1
    # load image
    src_name = src_path + name
    image = cv2.imread(src_name, cv2.IMREAD_GRAYSCALE)
    image_values.append(image)

    # DFT
    f = np.fft.fft2(image)
    fshift = np.abs(np.fft.fftshift(f))
    fshift_values.append(fshift)
    fshift = fshift*255/fshift.max()  # scale between 0 and 255
    dst_name = dst_path + name + '_dft.bmp'
    cv2.imwrite(dst_name, fshift)

    # DCT
    c = cv2.dct(np.float32(image))
    dct_values.append(c)
    c = c*255/c.max()  # scale between 0 and 255
    dst_name = dst_path + name + '_dct.bmp'
    cv2.imwrite(dst_name, c)

    # plot
    ax[idx,0].set_title('Original')
    ax[idx,0].imshow(image, cmap='gray')

    ax[idx,1].set_title('DFT')
    ax[idx,1].imshow(fshift, cmap = 'gray')

    # I just zero all negative value to make demonstration more comprehensive, but save images with negative values
    c[c<0] = 0
    ax[idx,2].set_title('DCT')
    ax[idx,2].imshow(c, cmap = 'gray')
    plt.savefig(f'ut{i}.jpg')
    i+=1