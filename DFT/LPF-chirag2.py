import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import math

import warnings
warnings.filterwarnings("ignore")

def transform(img, inp):
    img = rgb2gray(img)
    var = inp/2
    print(var)
    replaceValue = 1

    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    # dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)


    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)     # center

    if var < 0 or var*2 > rows:
        print(f"{inp}px er en ulovlig verdi!\nBlir til at man har igjen {rows-var}px")
        exit(1)

    #Definerer hvor mye av fouriertransformasjonen som skal fjernes
    height = int(img.shape[0])
    width = int(img.shape[1])
    newHeight = (height*var)/2
    newWidth = (width*var)/2
    newHeight = int(math.sqrt(width*height*var / (width/height)))
    newWidth = int(math.sqrt(width*height*var / (width/height)) * (width/height))
    y = math.sqrt(var)
    removeHeight = (height-newHeight)
    removeWidth = (width-newWidth)

    # #Utskrift til terminal
    print(f"height={height}, {newHeight=}")
    print(f"Width={width}, {newWidth=}")
    print(f"Original pixelverdi: {height*width}px")
    print(f"Komprimert {var*2}% pixelverdi: {newHeight*newWidth}px")
    print(f"Fjerner: h={removeHeight}px, og w={removeWidth}px")
    print(f"Forhold: {(newWidth/newHeight)} skal være lik {(width/height)}")
    print(f"Andel pixeler: {(newWidth*newHeight)/(height*width)}, skal være lik {var*2}")
    print(f"Skaleringsfaktor: {(width/newWidth)} skal være lik {(height/newHeight)}")




    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    # mask[crow-int(var/2):crow+int(var/2), ccol-int(var/2):ccol+int(var/2)] = replaceValue
    mask[-int((var/100)*rows):, :] = not replaceValue
    mask[:int((var/100)*rows), :] = not replaceValue
    mask[:, -int((var/100)*cols):] = not replaceValue
    mask[:,:int((var/100)*cols)] = not replaceValue
    if var == 0:
        mask[:, :] = replaceValue

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    masked_magnitude_spectrum = magnitude_spectrum.copy()
    # magnitude_spectrum[crow-var:crow+var, ccol-var:ccol+var] = replaceValue
    # masked_magnitude_spectrum[crow-int(var/2):crow+int(var/2), ccol-int(var/2):ccol+int(var/2)]
    masked_magnitude_spectrum[:int((var/100)*rows), :] = replaceValue
    masked_magnitude_spectrum[-int((var/100)*rows):, :] = replaceValue
    masked_magnitude_spectrum[:,:int((var/100)*cols)] = replaceValue
    masked_magnitude_spectrum[:, -int((var/100)*cols):] = replaceValue
    if var == 0:
        masked_magnitude_spectrum = magnitude_spectrum
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
     # # Statistikk
    pixelerIgjen = 0
    for k in range(masked_magnitude_spectrum.shape[0]):
        for j in range(masked_magnitude_spectrum.shape[1]):
            p = masked_magnitude_spectrum[k, j]
            if not p == i: 
                pixelerIgjen += 1
    print(f"{pixelerIgjen=}")

    return magnitude_spectrum, mask, masked_magnitude_spectrum, img_back

img = cv2.imread('Joddski.jpg')
# img = cv2.imread('ChiragBilde.jpg',0)
i = int(input("Prosent: "))

magnitude_spectrum, mask, masked_magnitude_spectrum, img_tr = transform(img, i)

plt.subplot(151)
plt.imshow(img, cmap='gray', interpolation="none", vmin=0)
plt.title('Input image'), plt.xticks([]), plt.yticks([])

plt.subplot(152)
plt.imshow(magnitude_spectrum, cmap='gray', interpolation="none", vmin=0, vmax=255)
plt.title('Magnitude\nspectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(153)
plt.imshow(masked_magnitude_spectrum, cmap='gray', interpolation="none", vmin=0, vmax=255)
plt.title('Masked\nmagnitude\nspectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(154)
plt.imshow(img_tr, cmap='gray', interpolation="none", vmin=0)
plt.title('Output image'), plt.xticks([]), plt.yticks([])

plt.subplot(155)
plt.imshow(mask[:,:,0], cmap='gray', interpolation="none", vmin=0, vmax=1)
plt.title('Mask'), plt.xticks([]), plt.yticks([])





plt.savefig(f'hei.jpg', format='jpeg', dpi=1200)     
# print(f"Lagret nr {i}")
