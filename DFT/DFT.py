import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Leser bildet i sort/hvit
img = cv2.imread('T.jpg', 0)

# Omgjør bildet til float32
imf = np.float32(img) 

# Finner dft av bildet 
dft = cv2.dft(imf, cv2.DFT_ROWS)
#dft = cv2.dft(imf, flags = cv2.DFT_COMPLEX_OUTPUT)

plt.hist(img.ravel(), bins=range(256), fc='k', ec='k')
plt.savefig("spekter.jpg")

# Omgjør bildet til array
img_arr = np.array(dft) 

# Klipper dft bildet
height = int(img_arr.shape[0])
width = int(img_arr.shape[1])
newheight = int(input(f'Tall mellom 0 og {height}: '))
newwidth = int(input(f'Tall mellom 0 og {width}: '))
img_arr[newheight:, :] = 0
img_arr[:, newwidth:] = 0

# Lagrer klippet dft bildet
img2 = Image.fromarray(img_arr).convert('RGB') 
img2.save("klippet1.jpg") 

# Finner den inverse discrete cosine transform av dft bildet
img1 = cv2.idft(img_arr)

# Omgjør bildet til uint8
img1 = np.uint8(img1)

# Lagrer bildet
all = np.concatenate ((img, img_arr, img1), axis = 1)
cv2.imwrite("DFT.jpg", dft)
cv2.imwrite("IDFT back image.jpg", all)