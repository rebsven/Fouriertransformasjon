import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



bildenavn = 'T.jpg'


def transform(img, inp):
    # img = cv2.imread(img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = inp

    replaceValue = 0

    img_float32 = np.float32(img)

    # Finner discrete cosine transform av bildet 
    dct = cv2.dct(img_float32, cv2.DCT_ROWS)

    # Omgjør bildet til array
    dct_arr = np.array(dct) 

    rows, cols = dct_arr.shape

    # create a mask first, center square is 1, remaining all zeros
    mask_arr = np.ones((rows, cols), np.uint8)
    mask_arr[-int(var):, :] = replaceValue
    mask_arr[:, -int(var):] = replaceValue
    if var == 0:
        mask_arr[:, :] = not replaceValue
    mask_img = Image.fromarray(mask_arr).convert('L')


    # Normal Magnitude spectrum
    freq_spectrum_image = Image.fromarray(dct_arr).convert('L')
    
    masked_freq_spectrum = dct_arr.copy()
    masked_freq_spectrum[-int(var):, :] = replaceValue 
    masked_freq_spectrum[:, -int(var):] = replaceValue 
    masked_freq_spectrum_image = Image.fromarray(masked_freq_spectrum).convert('L')
    if var == 0:
        masked_freq_spectrum = dct_arr
        masked_freq_spectrum_image = freq_spectrum_image

    # apply mask and inverse DCT
    img_back = Image.fromarray(cv2.idct(masked_freq_spectrum)).convert('RGB')
    # print(f"{mask_img=}\n{freq_spectrum_image=}\n{masked_freq_spectrum_image=}\n{img_back=}")



    return freq_spectrum_image, mask_img, masked_freq_spectrum_image, img_back

img = cv2.imread(bildenavn)
for i in range(25):
    if not i in [6, 8, 10]: # For specific tests
        continue

    freq_spectrum, mask, masked_freq_spectrum, img_tr = transform(img, i)

    
    vminV=np.min(np.array(img))
    vmaxV=np.max(np.array(img))

    f_size = 15
    fig, ax = plt.subplots(1,5,figsize=(f_size,f_size))
    
    ax[0].imshow(img, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    ax[0].set_title('Original', fontsize = f_size)

    
    ax[1].imshow(freq_spectrum, cmap='gray', interpolation="none", vmin=0)
    ax[1].set_title('DCT', fontsize = f_size)

    
    ax[2].imshow(mask, cmap='gray', interpolation="none", vmin=0)
    ax[2].set_title('Maske', fontsize = f_size)

    
    ax[3].imshow(masked_freq_spectrum, cmap='gray', interpolation="none", vmin=0)
    ax[3].set_title('DCT\nredusert', fontsize = f_size)

    
    ax[4].imshow(img_tr, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    ax[4].set_title('Komprimert\nprodukt', fontsize = f_size)
  
    try:
        plt.savefig(f'{bildenavn.split(".")[0]}/Komprimert_{i}.jpg', format='jpeg', dpi=1200)     
    except:
        print(f'Trenger mappenavn som er: {bildenavn.split(".")[0]}')
        exit()
    print(f"Lagret nr {i}")

    plt.clf()
