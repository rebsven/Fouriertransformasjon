
from PIL import Image 
import numpy as np       
  
# Opening the image and converting  
# it to RGB color mode 
# IMAGE_PATH => Path to the image 
img = Image.open(r"T.jpg").convert('RGB') 
  
# Extracting the image data & 
# creating an numpy array out of it 
img_arr = np.array(img) 


height = int(img_arr.shape[0])
width = int(img_arr.shape[1])
print(height)
print(width)

nr = int(input("Tall mellom 1 og 24: "))
newheigt = nr
newwith = nr

img_arr[newheigt:, :] = 0
img_arr[:, newwith:] = 0

# img_arr[0:0, :width] = (0,0,0)ss

# Turning the pixel values of the 400x400 pixels to black  
# img_arr[0: 125, 0 : 125] = (0, 0, 0) 
  
# Creating an image out of the previously modified array 
img = Image.fromarray(img_arr) 
  
# Displaying the image 
img.save("halv.jpg") 