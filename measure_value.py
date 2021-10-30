import numpy as np
import cv2
import math
from numpy.lib.type_check import imag


def dispay_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(img, down_points):
    return cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)


def zscore(img):
    return (img - np.mean(img))/np.std(img)


def clip(x, l, u):
    return l if x < l else u if x > u else x


def gaussian_kernel(img):

    width, height = np.shape(img)

    kernel = np.array([
        [2, 4,  5,  4,  2],
        [4, 9,  12, 9,  4],
        [5, 12, 15, 12, 5],
        [4, 9,  12, 9,  4],
        [2, 4,  6,  4,  2]
    ]) / 159

    kernel_width, kernel_height = np.shape(kernel)

    img_convoluted = np.zeros(
        (width-kernel_width+1, height-kernel_height+1))

    for i in range(np.shape(img_convoluted)[0]):
        for j in range(np.shape(img_convoluted)[1]):

            window_start_x, window_end_x = i, i+kernel_width
            window_start_y, window_end_y = j, j+kernel_height

            window = img[window_start_x:window_end_x,
                         window_start_y:window_end_y]

            img_convoluted[i, j] = np.sum(np.multiply(window, kernel))

    return img_convoluted


img = cv2.imread('19.2 capstone_coins.png', 0).astype(float)
img_size_new = (np.shape(img)[1]//2, np.shape(img)[0]//2)
img_resized = resize_image(img, img_size_new)

print(type(img_resized))
print(np.shape(img_resized))

img_resized_z = zscore(img_resized)
img_resized_z_guassian = gaussian_kernel(img_resized_z)

intensity_mat_x = (
    img_resized_z_guassian[2:][:] - img_resized_z_guassian[0:-2][:])/2
intensity_mat_y = (
    img_resized_z_guassian[:][2:] - img_resized_z_guassian[:][0:-2])/2

intensity_total = np.sqrt(intensity_mat_x**2 + intensity_mat_y**2)
gradient_total = np.arctan2(intensity_mat_y, intensity_mat_x)

# angles = np.array([0, math.pi/4, math.pi/2, math.pi*3/4])

intensity_total_z = zscore(intensity_total)

print(np.mean(intensity_total_z))
dispay_image(intensity_total_z)
