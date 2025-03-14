import cv2
from matplotlib import pyplot as plt
import numpy as np

# im = cv2.imread('attachments/Powisset-Documents/POW-wb-harvest-week-090423.jpg')
# imS = cv2.resize(im, (960, 540))
# cv2.imshow("output", imS)  

im = cv2.imread('/home/eddy/github.com/agro-doc/imseg/whiteboard_yellow_line.jpg')
# im_resized = cv2.resize(im, (960, 540))
# cv2.imshow('im', im_resized)



# Yellow Masking
im_r = cv2.resize(im, (960, 540))
im_hsv = cv2.cvtColor(im_r, cv2.COLOR_RGB2HSV)

lower_yellow = np.array([50, 120, 160])
upper_yellow = np.array([65, 255, 255])
mask = cv2.inRange(im_hsv, lower_yellow, upper_yellow) 
result = cv2.bitwise_and(im_r, im_r, mask = mask) 

# Show image
cv2.imshow('image', im_r) 
cv2.imshow('mask', mask) 
cv2.imshow('result', result) 

cv2.waitKey(0) 
cv2.destroyAllWindows() 
"""

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply edge detection method on the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
plt.subplot(121),plt.imshow(im,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
