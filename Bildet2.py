import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

dark_image = imread('test.jpg')
dark_image_grey = rgb2gray(dark_image)
x = int(input("Skriv inn prosentverid: "))
if (x > 100):
    print("Ikke gyldig verdi")
    exit()
prosentverdi = x/100


plt.hist(dark_image.ravel(), bins=range(256), fc='k', ec='k')
plt.savefig("Hello.jpg")


dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
fourier = dark_image_grey_fourier
plt.figure(num=None, figsize=(8, 6), dpi=200)
plt.imshow(np.log(abs(dark_image_grey_fourier)))
plt.savefig("out.jpg")
height = int(dark_image_grey_fourier.shape[0])
width = int(dark_image_grey_fourier.shape[1])
print(height)
print(width)


# h = sqrt(widthO*hO*%  /   h/width)
newHeight = int(math.sqrt(width*height*prosentverdi / (width/height)))

# width = sqrt(widthO*hO*%  /   h/width) * h/width
newWidth = int(math.sqrt(width*height*prosentverdi / (width/height)) * (width/height))

print(f"height={height}, {newHeight=}")
print(f"Width={width}, {newWidth=}")


y = math.sqrt(prosentverdi)
# z = newWidth # int(y*height)
# w = newHeight # int(y*width)

print(f"Original pixelverdi: {height*width}px")
print(f"Komprimert {x}% pixelverdi: {newHeight*newWidth}px")
removeHeight = (height-newHeight)
removeWidth = (width-newWidth)
print(f"Fjerner: h={removeHeight}px, og w={removeWidth}px")

print(f"Forhold: {(newWidth/newHeight)} skal være lik {(width/height)}")
print(f"Andel pixeler: {(newWidth*newHeight)/(height*width)}, skal være lik {prosentverdi}")
print(f"Skaleringsfaktor: {(width/newWidth)} skal være lik {(height/newHeight)}")



# print(y)
# print(z)
# print(w)
# print(height*width)
# print(w*z)
# print(int(((w*z)/(height*width))*100))


def fourier_masker_hor(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(image)))
    # Horisontal, full height
    delingsfaktor = 2
    dark_image_grey_fourier[:int(removeWidth/delingsfaktor), :] = i 
    dark_image_grey_fourier[-int(removeWidth/delingsfaktor):, :] = i
    # Vertikal, full width
    dark_image_grey_fourier[:, :int(removeHeight/delingsfaktor)] = i 
    dark_image_grey_fourier[:, -int(removeHeight/delingsfaktor):] = i
    # dark_image_grey_fourier[:height, 0:int(w/4)] = i
    # dark_image_grey_fourier[-height:, int(width-(w/4)):width] = i
    # Statistikk
    pixelerIgjen = 0
    for k in range(dark_image_grey_fourier.shape[0]):
        for j in range(dark_image_grey_fourier.shape[1]):
            p = dark_image_grey_fourier[k, j]
            #print(p)
            #break
            if not p == i: 
                pixelerIgjen += 1
    print(f"{pixelerIgjen=}")


    fig, ax = plt.subplots(1,4,figsize=(15,15))
    ax[0].imshow(rgb2gray(image), cmap = 'gray')
    ax[0].set_title('Orginalbildet', fontsize = f_size)
    ax[1].imshow(np.log(abs(fourier)), cmap='gray')
    ax[1].set_title('Fouriertransformajon', fontsize = f_size)
    ax[2].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[2].set_title('Fouriertransformasjonen\nredusert', fontsize = f_size)
    ax[3].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), cmap='gray')
    ax[3].set_title('Transformert bildet', fontsize = f_size)
    return plt

plt = fourier_masker_hor(dark_image, (0+1j)) # 0.0000000000000001
plt.savefig("hor.jpg")


 
