import cv2
import matplotlib.pyplot as plt
import numpy as np

# Leser bildet i sort/hvit
img = cv2.imread('test.jpg', 0)

# Check if the image is not quadratic
if img.shape[0] != img.shape[1]:
    # Get the size of the smallest side
    size = min(img.shape[0], img.shape[1])
    
    # Create a black image of the size of the smallest side
    black_img = np.zeros((size, size, 3), np.uint8)
    
    # Copy the image to the black image, centered
    black_img[(size - img.shape[0])//2:(size - img.shape[0])//2 + img.shape[0], 
              (size - img.shape[1])//2:(size - img.shape[1])//2 + img.shape[1]] = img
    
    # Set the black_img as the new image
    img = black_img

# Omgjør bildet til float32
imf = np.float32(img) 

# Finner dft av bildet 
dft = cv2.dft(imf, cv2.DFT_ROWS)

plt.hist(img.ravel(), bins=range(256), fc='k', ec='k')
plt.savefig("spekter.jpg")

# Omgjør bildet til array
img_arr = np.array(dft) 

# Shift quadrants
img_arr_shift = np.zeros_like(img_arr)
img_arr_shift[0:img_arr.shape[0]//2, 0:img_arr.shape[1]//2] = img_arr[img_arr.shape[0]//2:, img_arr.shape[1]//2:]
img_arr_shift[0:img_arr.shape[0]//2, img_arr.shape[1]//2:] = img_arr[img_arr.shape[0]//2:, 0:img_arr.shape[1]//2]
img_arr_shift[img_arr.shape[0]//2:, 0:img_arr.shape[1]//2] = img_arr[0:img_arr.shape[0]//2, img_arr.shape[1]//2:]
img_arr_shift[img_arr.shape[0]//2:, img_arr.shape[1]//2:] = img_arr[0:img_arr.shape[0]//2, 0:img_arr.shape[1]//2]

# Finner den inverse discrete cosine transform av dft bildet
img1 = cv2.idft(img_arr_shift)

# Omgjør bildet til uint8
img1 = np.uint8(img1)

# Lagrer bildet
all = np.concatenate ((img, img_arr_shift, img1), axis = 1)
cv2.imwrite("DFT.jpg", dft)
cv2.imwrite("IDFT back image.jpg", all)