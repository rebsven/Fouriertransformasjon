import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



bildenavn = 'T.jpg'


def transform(img, inp):
    # img = cv2.imread(img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = inp/10

    replaceValue = 0

    img_float32 = np.float32(img)

    # Finner discrete cosine transform av bildet 
    dct = cv2.dct(img_float32, cv2.DCT_ROWS)

    # Omgjør bildet til array
    dct_arr = np.array(dct) 

    rows, cols = dct_arr.shape

    # create a mask first, center square is 1, remaining all zeros
    mask_arr = np.ones((rows, cols), np.uint8)
    mask_arr[-int(var*rows):, :] = replaceValue
    mask_arr[:, -int(var*cols):] = replaceValue
    if var == 0:
        mask_arr[:, :] = not replaceValue
    mask_img = Image.fromarray(mask_arr).convert('L')
    mask_img.save("0_mask.jpg")


    # Normal Magnitude spectrum
    freq_spectrum_image = Image.fromarray(dct_arr).convert('L')
    freq_spectrum_image.save("0_freq.jpg")
    
    masked_freq_spectrum = dct_arr
    masked_freq_spectrum[-int(var*rows):, :] = replaceValue 
    masked_freq_spectrum[:, -int(var*cols):] = replaceValue 
    masked_freq_spectrum_image = Image.fromarray(masked_freq_spectrum).convert('L')
    if var == 0:
        masked_freq_spectrum_image = freq_spectrum_image
    masked_freq_spectrum_image.save("0_masked_freq.jpg")


    # apply mask and inverse DCT
    # img_back = np.uint8(cv2.idct(masked_freq_spectrum))
    img_back = Image.fromarray(cv2.idct(masked_freq_spectrum)).convert('RGB')
    # print(f"{mask_img=}\n{freq_spectrum_image=}\n{masked_freq_spectrum_image=}\n{img_back=}")

    img_back.save("0_fin.jpg")


    return freq_spectrum_image, mask_img, masked_freq_spectrum_image, img_back

img = cv2.imread(bildenavn)
for i in range(11):
    # if not i in [0, 1, 5, 10]:
    #     continue

    freq_spectrum, mask, masked_freq_spectrum, img_tr = transform(img, i)

    print(type(bildenavn))

    plt.subplot(151)
    plt.imshow(img, cmap='gray', interpolation="none", vmin=0)
    plt.title('Input image'), plt.xticks([]), plt.yticks([])

    plt.subplot(152)
    plt.imshow(freq_spectrum, cmap='gray', interpolation="none", vmin=0)
    plt.title('Magnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(153)
    plt.imshow(mask, cmap='gray', interpolation="none")
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(154)
    plt.imshow(masked_freq_spectrum, cmap='gray', interpolation="none", vmin=0)
    plt.title('Masked\nmagnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(155)
    plt.imshow(img_tr, cmap='gray', interpolation="none", vmin=0)
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    
    try:
        plt.savefig(f'{bildenavn.split(".")[0]}/Komprimert_{i}0_prosent.jpg', format='jpeg', dpi=1200)     
    except:
        print(f'Trenger mappenavn som er: {bildenavn.split(".")[0]}')
        exit()
    print(f"Lagret nr {i}")
