import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


B=1600 #blocksize
fn3='test2.jfif'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
print(img1)
# help(cv2.imread)
h,w = np.array(img1.shape[:2])/B * B
#h,w = np.array(img1.shape[:2])
h = int(h)
w = int(w)
print (h)
print (w)
img1=img1[:int(h),:int(w)]


blocksV=h/B
blocksH=w/B
print (blocksH)
print (blocksV)
vis0 = np.zeros((h,w), np.float32)
Trans = np.zeros((h,w), np.float32)
vis0[:h, :w] = img1
for row in range(int(blocksV)):
        for col in range(int(blocksH)):
                currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
# cv2.imwrite('Transformed.jpg', cv2.putText(Trans, "ok", (0,0), 1, 12, 0))
cv2.imwrite('Transformed.jpg', Trans)


# plt.savefig(img1)

point=plt.ginput(1,10)
block=np.floor(np.array(point)/B) #first component is col, second component is row
print (block)
col=block[0,0]
row=block[0,1]
plt.plot([B*col,B*col+B,B*col+B,B*col,B*col],[B*row,B*row,B*row+B,B*row+B,B*row])
plt.axis([0,w,h,0])
plt.title("Original Image")

plt.figure()
plt.subplot(1,2,1)
selectedImg=img1[row*B:(row+1)*B,col*B:(col+1)*B]
N255=Normalize(0,255) #Normalization object, used by imshow()
plt.title("Image in selected Region")
plt.imshow(selectedImg,cmap="gray",norm=N255,interpolation='nearest')
plt.savefig("Heye.jpg")


plt.subplot(1,2,2)
selectedTrans=Trans[row*B:(row+1)*B,col*B:(col+1)*B]
plt.imshow(selectedTrans,cmap='gray',interpolation='nearest')
plt.colorbar(shrink=0.5)
plt.title("DCT transform of selected Region")
plt.savefig("Heee.jpg")

back0 = np.zeros((h,w), np.float32)
for row in range(int(blocksV)):
        for col in range(int(blocksH)):
                currentblock = cv2.idct(Trans[row*B:(row+1)*B,col*B:(col+1)*B])
                back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
cv2.imwrite('BackTransformed.jpg', back0)

diff=back0-img1
print (diff.max())
print (diff.min())
MAD=np.sum(np.abs(diff))/float(h*w)
print ("Mean Absolute Difference: ",MAD)
plt.figure()
plt.imshow(back0,cmap="gray")
plt.title("Backtransformed Image")
plt.savefig("Hoy.jpg")