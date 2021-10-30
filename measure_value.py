import numpy as np
import cv2
import matplotlib.pyplot as plt

# reads image 'opencv-logo.png' as grayscale
img = cv2.imread('19.2 capstone_coins.png')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
