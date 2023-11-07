import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import math 


# Leser bildet i sort/hivt
img = cv2.imread('Joddski.jpg', 0)

# Omgjør bildet til float32
imf = np.float32(img) 

# Finenr discrete cosine transform av bildet
dct = cv2.dct(imf, cv2.DCT_ROWS)



# Omgjør til et array
img_arr = np.array(dct) 

# # Klipper dct bildet
# height = int(img_arr.shape[0])
# width = int(img_arr.shape[1])
nr = int(input("Prosent: "))
nr = nr/100
# newheigt = int(height*nr)
# newwidth = int(width*nr)
# img_arr[newheigt:, :] = 0
# img_arr[:, newwidth:] = 0

#Definerer hvor mye av fouriertransformasjonen som skal fjernes
height = int(img_arr.shape[0])
width = int(img_arr.shape[1])
newHeight = (height*nr)
newWidth = (width*nr)
# newHeight = int(math.sqrt(width*height*nr / (width/height)))
# newWidth = int(math.sqrt(width*height*nr / (width/height)) * (width/height))
# # y = math.sqrt(nr)
# removeHeight = (height-newHeight)
# removeWidth = (width-newWidth)
print(newWidth)
print(newHeight)
pixelerIgjen = 0
for k in range(img_arr.shape[0]):
    for j in range(img_arr.shape[1]):
        p = img_arr[k, j]
        if not p == 1: 
            pixelerIgjen += 1
print(f"{pixelerIgjen=}")

delingsfaktor = 1
img_arr[int(newHeight/delingsfaktor):, :] = 1 #int(removeHeight/delingsfaktor)
# img_arr[-int(removeHeight/delingsfaktor):, :] = 1
# Vertikal, full width
# img_arr[:, :int(removeWidth/delingsfaktor)] = 1 #int(removeWidth/delingsfaktor)
img_arr[:, int(newWidth/delingsfaktor):] = 1

pixelerIgjen = 0
for k in range(img_arr.shape[0]):
    for j in range(img_arr.shape[1]):
        p = img_arr[k, j]
        if not p == 1: 
            pixelerIgjen += 1
print(f"{pixelerIgjen=}")
print(newWidth)
print(newHeight)
# Lagrer det klippte dct bildet
img2 = Image.fromarray(img_arr).convert('L') 
img2.save("klippetDCT.jpg") 


# Finner den inverse discrete cosine transform av dct bildet
img1 = cv2.idct(img_arr)

# Omgjør bilet til uint8
img1 = np.uint8(img1)

plt.subplot(141)
plt.imshow(img, cmap='gray', interpolation="none", vmin=0, vmax=255)
plt.title('Input image'), plt.xticks([]), plt.yticks([])

plt.subplot(142)
plt.imshow(dct, cmap='gray', interpolation="none", vmin=0, vmax=255)
plt.title('Magnitude\nspectrum'), plt.xticks([]), plt.yticks([])
# print(f"Verdi for [2, 2] er {magnitude_spectrum[2, 2]}")

plt.subplot(143)
plt.imshow(img_arr, cmap='gray', interpolation="none", vmin=0, vmax=1)
plt.title('Mask'), plt.xticks([]), plt.yticks([])

plt.subplot(144)
plt.imshow(img1, cmap='gray', interpolation="none", vmin=0, vmax=255)
plt.title('Masked\nmagnitude\nspectrum'), plt.xticks([]), plt.yticks([])

plt.hist(dct.ravel(), bins=range(256), fc='k', ec='k')
plt.savefig("DCTtest.jpg", format='jpeg', dpi=1200)
# Setter sammen bildet
all = np.concatenate ((img, img_arr, img1), axis = 1)
cv2.imwrite("DCT.jpg", dct)
cv2.imwrite("IDCT back image.jpg", all)