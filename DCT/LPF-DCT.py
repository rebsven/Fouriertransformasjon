import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



bildenavn = 'T.jpg'


def transform(img, inp):
    var = inp/10

    replaceValue = 0
    convertMode = 'L'
    scale = 255.0

    img_float32 = np.float32(img)# / (scale))

    # Finner discrete cosine transform av bildet 
    dct = cv2.dct(img_float32, cv2.DCT_ROWS)
    cv2.imwrite("0_freq.jpg", dct)

    # Omgjør bildet til array
    dct_arr = np.array(dct) 
    # dct_arr = dct 

    rows, cols = dct_arr.shape

    # create a mask first, to show area
    mask_arr = np.ones((rows, cols), np.uint8)
    mask_arr[-int(var*rows):, :] = replaceValue
    mask_arr[:, -int(var*cols):] = replaceValue
    if var == 0:
        mask_arr[:, :] = not replaceValue
    mask_img = Image.fromarray(mask_arr).convert('L')
    mask_img.save("0_mask.jpg")


    # Normal Magnitude spectrum
    freq_spectrum_image = Image.fromarray(np.uint8(np.log(abs(dct_arr))*scale)).convert(convertMode)
    if i == 0:
        # freq_spectrum_image.save("0_freq.jpg")
        pass
    
    masked_freq_spectrum_image = Image.fromarray(np.uint8(np.log(abs(dct_arr))*mask_arr*scale)).convert(convertMode)
    if var == 0:
        masked_freq_spectrum_image = freq_spectrum_image

    masked_freq_spectrum_image.save("0_masked_freq.jpg")
    # cv2.imwrite("0_masked_freq.jpg", masked_freq_spectrum_image)


    # apply mask and inverse DCT
    # img_back = np.uint8(cv2.idct(dct_arr*mask_arr))
    tmpDCT = cv2.idct(dct_arr*mask_arr)
    img_back = Image.fromarray(np.uint8(tmpDCT*scale)).convert(convertMode)
    # print(f"{mask_img=}\n{freq_spectrum_image=}\n{dct_arr*mask_arr=}\n{img_back=}")
    # print(f"{dct_arr[0,0]=}\n{dct_arr[0,1]=}\n{dct_arr[0,2]=}")
    # print(f"{dct_arr*mask_arr[0,0]=}\n{dct_arr*mask_arr[0,1]=}\n{dct_arr*mask_arr[0,2]=}")
    # print(f"{np.array(img_back)[0,0]=}\n{np.array(img_back)[0,1]=}\n{np.array(img_back)[0,2]=}")
    if inp in [2, 3, 4]:
        print(f"{np.array(img_back)[int(rows/2),int(cols/2)]=}")
    img_back.save("0_fin.jpg")


    return freq_spectrum_image, mask_img, masked_freq_spectrum_image, img_back

img = cv2.imread(bildenavn, 0)
for i in range(11):
    # if not i in [0, 1, 5, 10]:
    #     continue

    freq_spectrum, mask, masked_freq_spectrum, img_tr = transform(img, i)
    vminV=np.min(np.array(freq_spectrum))
    vmaxV=np.max(np.array(freq_spectrum))

    plt.subplot(151)
    plt.imshow(img, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    plt.title('Input image'), plt.xticks([]), plt.yticks([])

    plt.subplot(152)
    plt.imshow(freq_spectrum, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    plt.title('Magnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(153)
    plt.imshow(mask, cmap='gray', interpolation="none")
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(154)
    plt.imshow(masked_freq_spectrum, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    plt.title('Masked\nmagnitude\nspectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(155)
    plt.imshow(img_tr, cmap='gray', interpolation="none", vmin=vminV, vmax=vmaxV)
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    
    try:
        plt.savefig(f'{bildenavn.split(".")[0]}/Komprimert_{i}0_prosent.jpg', format='jpeg', dpi=1200)     
    except:
        print(f'Trenger mappenavn som er: {bildenavn.split(".")[0]}')
        exit()
    print(f"Lagret nr {i}")
