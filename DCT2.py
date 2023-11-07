# import required libraries
import cv2
import numpy as np

# read input image as grayscale
img = cv2.imread('T.png', 0)

# convert the grayscale to float32
imf = np.float32(img) # float conversion

# find discrete cosine transform
dst = cv2.dct(imf, cv2.DCT_INVERSE)
dst2 = dst

# apply inverse discrete cosine transform
img1 = cv2.idct(dst)

# convert to uint8
img1 = np.uint8(img)

height = int(dst2.shape[0])
width = int(dst2.shape[1])
print(height)
print(width)

valg = 5
newheigt = int(height*(valg/10))
newwith = int(width*(valg/10))

dst2[0 : 400, 0 : 400] = 0.0000000001

# apply inverse discrete cosine transform
img2 = cv2.idct(dst2)

# convert to uint8
img2 = np.uint8(img)

# display the images
cv2.imwrite("DCT.jpg", dst)
# cv2.waitKey(0)
cv2.imwrite("IDCT back image.jpg", img1)
cv2.imwrite("DCT2.jpg", dst2)
# cv2.waitKey(0)
cv2.imwrite("IDCT back image2.jpg", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()