import numpy as np
import cv2
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def transform(img, inp):        
    var = inp

    replaceValue = 1

    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)


    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)     # center

    if var < 0 or var*2 > rows:
        print(f"{inp}px er en ulovlig verdi!\nBlir til at man har igjen {rows-var}px")
        exit(1)


    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    # mask[crow-int(var/2):crow+int(var/2), ccol-int(var/2):ccol+int(var/2)] = replaceValue
    mask[-var:, :] = not replaceValue
    mask[:var, :] = not replaceValue
    mask[:, -var:] = not replaceValue
    mask[:,:var] = not replaceValue
    if var == 0:
        mask[:, :] = replaceValue

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    masked_magnitude_spectrum = magnitude_spectrum.copy()
    # magnitude_spectrum[crow-var:crow+var, ccol-var:ccol+var] = replaceValue
    # masked_magnitude_spectrum[crow-int(var/2):crow+int(var/2), ccol-int(var/2):ccol+int(var/2)]
    masked_magnitude_spectrum[:var, :] = replaceValue
    masked_magnitude_spectrum[-var:, :] = replaceValue
    masked_magnitude_spectrum[:,:var] = replaceValue
    masked_magnitude_spectrum[:, -var:] = replaceValue
    if var == 0:
        masked_magnitude_spectrum = magnitude_spectrum
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return magnitude_spectrum, mask, masked_magnitude_spectrum, img_back

img = cv2.imread('T.jpg',0)
for i in range(13):
    # if not i in [0, 1, 5, 10]:
    #     continue

    magnitude_spectrum, mask, masked_magnitude_spectrum, img_tr = transform(img, i)

    plt.subplot(151)
    plt.imshow(img, cmap='gray', interpolation="none", vmin=0)
    plt.title('Input image'), plt.xticks([]), plt.yticks([])

    plt.subplot(152)
    plt.imshow(magnitude_spectrum, cmap='gray', interpolation="none", vmin=0, vmax=255)
    plt.title('Magnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(153)
    plt.imshow(mask[:,:,0], cmap='gray', interpolation="none", vmin=0, vmax=1)
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(154)
    plt.imshow(masked_magnitude_spectrum, cmap='gray', interpolation="none", vmin=0, vmax=255)
    plt.title('Masked\nmagnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(155)
    plt.imshow(img_tr, cmap='gray', interpolation="none", vmin=0)
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    
    plt.savefig(f'moro/DFT2_{i}.jpg', format='jpeg', dpi=1200)     
    print(f"Lagret nr {i}")
