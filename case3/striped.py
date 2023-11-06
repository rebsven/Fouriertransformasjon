import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



bildenavn = 'T.jpg'


def transform(img):
    # img = cv2.imread(img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    replaceValue = 0

    img_float32 = np.float32(img)

    # Finner discrete cosine transform av bildet 
    dct = cv2.dct(img_float32, cv2.DCT_ROWS)

    # Omgjør bildet til array
    dct_arr = np.array(dct) 

    rows, cols = dct_arr.shape

    # create a mask first, center square is 1, remaining all zeros
    mask_arr = np.ones((rows, cols), np.uint8)
    for i in range(cols):
        if i % 2:
            mask_arr[:, i:i+1] = replaceValue
    mask_arr[12:13, :] = replaceValue
    mask_img = Image.fromarray(mask_arr).convert('L')


    # Normal Magnitude spectrum
    freq_spectrum_image = Image.fromarray(dct_arr).convert('L')
    
    masked_freq_spectrum = dct_arr.copy()
    for i in range(cols):
        if i % 2:
            masked_freq_spectrum[:, i:i+1] = replaceValue
    masked_freq_spectrum[12:13, :] = replaceValue
    masked_freq_spectrum_image = Image.fromarray(masked_freq_spectrum).convert('L')

    # apply mask and inverse DCT
    img_back = Image.fromarray(cv2.idct(masked_freq_spectrum)).convert('RGB')
    # print(f"{mask_img=}\n{freq_spectrum_image=}\n{masked_freq_spectrum_image=}\n{img_back=}")



    return freq_spectrum_image, mask_img, masked_freq_spectrum_image, img_back

img = cv2.imread(bildenavn)

freq_spectrum, mask, masked_freq_spectrum, img_tr = transform(img)


vminV=np.min(np.array(img))
vmaxV=np.max(np.array(img))

plt.subplot(151)
plt.imshow(img, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
plt.title('Input'), plt.xticks([]), plt.yticks([])

plt.subplot(152)
plt.imshow(freq_spectrum, cmap='gray', interpolation="none", vmin=0)
plt.title('DCT'), plt.xticks([]), plt.yticks([])

plt.subplot(153)
plt.imshow(mask, cmap='gray', interpolation="none", vmin=0, vmax=1)
plt.title('Mask'), plt.xticks([]), plt.yticks([])

plt.subplot(154)
plt.imshow(masked_freq_spectrum, cmap='gray', interpolation="none", vmin=0)
plt.title('Masked DCT'), plt.xticks([]), plt.yticks([])

plt.subplot(155)
plt.imshow(img_tr, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
plt.title('Output'), plt.xticks([]), plt.yticks([])

plt.savefig(f'Striped.jpg', format='jpeg', dpi=1200)     
print(f"Lagret")

plt.clf()
