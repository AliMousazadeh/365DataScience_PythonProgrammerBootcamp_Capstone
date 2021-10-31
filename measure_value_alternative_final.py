import numpy as np
import cv2
from numpy.lib.type_check import imag
import matplotlib.pyplot as plt


def zscore(img):
    return (img - np.mean(img))/np.std(img)


file_name = '19.2 capstone_coins.png'
reduction_ratio = 1

image_gray = cv2.imread(file_name, 0)
image_color = cv2.imread(file_name, 1)

dim = (np.shape(image_gray)[1] // reduction_ratio,
       np.shape(image_gray)[0] // reduction_ratio)

image_gray = cv2.resize(image_gray, dim, interpolation=cv2.INTER_LINEAR)
image_color = cv2.resize(image_color, dim, interpolation=cv2.INTER_LINEAR)


image_blurred = cv2.GaussianBlur(image_gray, (11, 11), 0)


f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(zscore(image_gray), cmap='gray')
axarr[0, 1].imshow(image_color)
axarr[1, 0].imshow(zscore(image_blurred), cmap='gray')

circles = cv2.HoughCircles(image_blurred, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=7, param2=70, minRadius=60, maxRadius=250)

if circles is not None:
    circles = np.round(circles[0, :]).astype('int')

    for (x, y, r) in circles:
        cv2.circle(image_color, (x, y), r, (0, 255, 0), 3)

axarr[1, 1].imshow(image_color)
plt.show()

radius_of_coins_to_find = np.round(
    np.array([88, 104, 124, 134, 140]) * reduction_ratio)
value_of_coins = [5, 1, 10, 2, 50]

radius_of_coins_found = circles[:, -1]
print(radius_of_coins_found)

radius_of_coins_found = np.reshape(
    radius_of_coins_found, (np.shape(radius_of_coins_found)[0], 1))
radius_of_coins_to_find = np.reshape(
    radius_of_coins_to_find, (np.shape(radius_of_coins_to_find)[0], 1))

distance_matrix = (radius_of_coins_found -
                   np.transpose(radius_of_coins_to_find))**2
coin_types = np.argmin(distance_matrix, axis=1)
print(coin_types)

total_value = 0
for i in range(np.shape(coin_types)[0]):
    total_value += value_of_coins[coin_types[i]]
print(total_value)
