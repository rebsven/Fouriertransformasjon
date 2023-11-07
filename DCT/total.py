import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 


# Leser bildet i sort/hvit
img = cv2.imread('T.jpg', 0)


# Omgjør bildet til float32
imf = np.float32(img) 

# Finner discrete cosine transform av bildet 
dct = cv2.dct(imf, cv2.DCT_ROWS)

# Omgjør bildet til array
img_arr = np.array(dct) 

# Klipper dct bildet
height = int(img_arr.shape[0])
width = int(img_arr.shape[1])
newheigt = int(input(f'Tall mellom 0 og {height}: '))
newwith = int(input(f'Tall mellom 0 og {width}: '))
img_arr[newheigt:, :] = 0
img_arr[:, newwith:] = 0

# Lagrer klippet dct bildet
img2 = Image.fromarray(img_arr).convert('RGB') 
img2.save("klippet.jpg") 

# Finner den inverse discrete cosine transform av dct bildet
img1 = cv2.idct(img_arr)

# Omgjør bildet til uint8
img1 = np.uint8(img1)

# Lagrer bildet
cv2.imwrite("DCT.jpg", dct)
cv2.imwrite("IDCT back image.jpg", img1)