import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
import warnings
warnings.filterwarnings("ignore")

#Leser inn bildet
dark_image = imread('ChiragBilde.jpg', 0) #Dhoni-dive_165121_730x419-m
dark_image_grey = rgb2gray(dark_image)

#Sjekker om input verdi er gryldig
x = int(input("Skriv inn prosentverid: "))
if (x > 100 or x < 0):
    print("Ikke gyldig verdi")
    exit()
prosentverdi = (x/100)

#Lager bildet av fargesppektert     Finn utt hvilke verdier det er på aksene!!!!!!!!!!!!!!!!!
plt.hist(dark_image.ravel(), bins=range(256), fc='k', ec='k')
plt.savefig("spekter.jpg")

#Gjør fft på bildet
dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
fourier = dark_image_grey_fourier
plt.figure(num=None, figsize=(8, 6), dpi=200)
plt.imshow(np.log(abs(dark_image_grey_fourier)))
plt.savefig("transforamsjon.jpg")

#Definerer hvor mye av fouriertransformasjonen som skal fjernes
height = int(dark_image_grey_fourier.shape[0])
width = int(dark_image_grey_fourier.shape[1])
newHeight = (height*prosentverdi)/2
newWidth = (width*prosentverdi)/2
newHeight = int(math.sqrt(width*height*prosentverdi / (width/height)))
newWidth = int(math.sqrt(width*height*prosentverdi / (width/height)) * (width/height))
y = math.sqrt(prosentverdi)
removeHeight = (height-newHeight)
removeWidth = (width-newWidth)

# #Utskrift til terminal
print(f"height={height}, {newHeight=}")
print(f"Width={width}, {newWidth=}")
print(f"Original pixelverdi: {height*width}px")
print(f"Komprimert {x}% pixelverdi: {newHeight*newWidth}px")
print(f"Fjerner: h={removeHeight}px, og w={removeWidth}px")
print(f"Forhold: {(newWidth/newHeight)} skal være lik {(width/height)}")
print(f"Andel pixeler: {(newWidth*newHeight)/(height*width)}, skal være lik {prosentverdi}")
print(f"Skaleringsfaktor: {(width/newWidth)} skal være lik {(height/newHeight)}")



def fourier_masker(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))

    #Reuserer foriertransformasjonen med ønsket verdi
    # Horisontal, full height
    delingsfaktor = 2
    dark_image_grey_fourier[:int(removeHeight/delingsfaktor), :] = i #int(removeHeight/delingsfaktor)
    dark_image_grey_fourier[-int(removeHeight/delingsfaktor):, :] = i
    # Vertikal, full width
    dark_image_grey_fourier[:, :int(removeWidth/delingsfaktor)] = i #int(removeWidth/delingsfaktor)
    dark_image_grey_fourier[:, -int(removeWidth/delingsfaktor):] = i

    # dark_image_grey_fourier[:int(newHeight), :] = i 
    # dark_image_grey_fourier[-int(newHeight):, :] = i
    # # Vertikal, full width
    # dark_image_grey_fourier[:, :int(newWidth)] = i 
    # dark_image_grey_fourier[:, -int(newWidth):] = i

    # # Statistikk
    pixelerIgjen = 0
    for k in range(dark_image_grey_fourier.shape[0]):
        for j in range(dark_image_grey_fourier.shape[1]):
            p = dark_image_grey_fourier[k, j]
            if not p == i: 
                pixelerIgjen += 1
    print(f"{pixelerIgjen=}")

    vminV=np.min(np.array(image))
    vmaxV=np.max(np.array(image))

    # Plotter ferdig bildet
    fig, ax = plt.subplots(1,4,figsize=(15,15))
    ax[0].imshow(image, cmap = 'gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    ax[0].set_title('Orginalbildet', fontsize = f_size)
    ax[1].imshow(np.log(abs(fourier)), cmap='gray', interpolation="none", vmin=0)
    ax[1].set_title('Fouriertransformajon', fontsize = f_size)
    ax[2].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray', interpolation="none", vmin=0)
    ax[2].set_title('Fouriertransformasjonen\nredusert', fontsize = f_size)
    ax[3].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), cmap='gray',  interpolation="none", vmin=vminV, vmax=vmaxV)
    ax[3].set_title('Transformert bildet', fontsize = f_size)
    return plt

plt = fourier_masker(dark_image_grey, 1j) # 0.0000000000000001
plt.savefig("ut.jpg", format='jpeg', dpi=1200)


 
