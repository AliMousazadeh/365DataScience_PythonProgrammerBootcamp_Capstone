import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from numpy import ma
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


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def draw_circle(grid, center, radius):
    width, height = np.shape(grid)
    T = np.arange(0, 2*math.pi, 2*math.pi/100)
    for t in T:
        x_idx = round(clip(center[0] + radius*math.cos(t), 0, width-1))
        y_idx = round(clip(center[1] + radius*math.sin(t), 0, height-1))
        grid[x_idx, y_idx] = 1
    return grid


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


reduction_ratio = 3
img = cv2.imread('19.2 capstone_coins.png', 0).astype(float)
img_size_new = (np.shape(img)[1]//reduction_ratio,
                np.shape(img)[0]//reduction_ratio)
img_resized = resize_image(img, img_size_new)

print(type(img_resized))
print(np.shape(img_resized))

# print(np.max(img_resized))
img_resized_z = zscore(img_resized)
# img_resized_z = (img_resized) / 255
# img_resized_z = img_resized
img_resized_z_guassian = gaussian_kernel(img_resized_z)

intensity_mat_x = (
    img_resized_z_guassian[2:][:] - img_resized_z_guassian[0:-2][:])/2
intensity_mat_y = (
    img_resized_z_guassian[:][2:] - img_resized_z_guassian[:][0:-2])/2

intensity_total = np.sqrt(intensity_mat_x**2 + intensity_mat_y**2)
gradient_total = np.arctan2(intensity_mat_y, intensity_mat_x)

intensity_total_pre_grad = np.copy(intensity_total)

angles = np.array([0, math.pi/4, math.pi/2, math.pi*3/4])

for i in range(1, np.shape(intensity_total)[0]-1):
    for j in range(1, np.shape(intensity_total)[1]-1):
        index = np.fabs(np.fabs(gradient_total[i, j]) - angles) < 0.1

        if index[0]:
            if not(intensity_total[i, j] > intensity_total[i+1, j] and intensity_total[i, j] > intensity_total[i-1, j]):
                intensity_total[i, j] = 0
        elif index[1]:
            if not(intensity_total[i, j] > intensity_total[i+1, j+1] and intensity_total[i, j] > intensity_total[i-1, j-1]):
                intensity_total[i, j] = 0
        elif index[2]:
            if not(intensity_total[i, j] > intensity_total[i, j+1] and intensity_total[i, j] > intensity_total[i, j-1]):
                intensity_total[i, j] = 0
        elif index[3]:
            if not(intensity_total[i, j] > intensity_total[i-1, j+1] and intensity_total[i, j] > intensity_total[i+1, j-1]):
                intensity_total[i, j] = 0
        else:
            print('threshold too low')


mu, st = np.mean(intensity_total), np.std(intensity_total)
min_intensity, max_intensity = np.min(intensity_total), np.max(intensity_total)
print(mu, st, min_intensity, max_intensity)

intensity_total_grad = np.copy(intensity_total)

pixel_status = np.zeros(np.shape(intensity_total))
pixel_status[intensity_total > 20*mu] = 1
pixel_status[intensity_total < 5*mu] = -1

pixel_status_memory = np.copy(pixel_status)
strong_points = []
for i in range(1, np.shape(pixel_status)[0]-1):
    for j in range(1, np.shape(pixel_status)[1]-1):
        someone_stronger = np.sum(
            (pixel_status[i-1:i+2, j-1:j+2] > 0).astype(int))

        if pixel_status[i, j] == -1:
            if someone_stronger:
                pixel_status_memory[i, j] = 1
                strong_points.append([i, j])
            else:
                intensity_total[i, j] = 0
        elif pixel_status[i, j] == 1:
            strong_points.append([i, j])

pixel_status = np.copy(pixel_status_memory)

intensity_total_strong = np.copy(intensity_total)
intensity_total_strong_final = (intensity_total_strong > 0).astype(int)

rmse = np.sqrt(np.mean((intensity_total_grad-intensity_total_strong)**2))
print(rmse)

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(zscore(intensity_total_pre_grad), cmap='gray')
axarr[0, 1].imshow(zscore(intensity_total_grad), cmap='gray')
axarr[1, 0].imshow(zscore(intensity_total_strong), cmap='gray')
axarr[1, 1].imshow(zscore(intensity_total_strong_final), cmap='gray')
plt.show()


print('Counting the votes...')
W, H = np.shape(intensity_total)
radius_of_coins = np.round(np.array([48, 55, 67, 71]) * 2/reduction_ratio)
number_of_coins = [1, 4, 1, 2]
value_of_coins = [5, 1, 10, 2]
T = np.arange(0, 2*math.pi, 2*math.pi/100)
votes_total = np.zeros((W, H))
circles = {}
for j in range(len(radius_of_coins)):
    votes_for_radius = np.zeros((W, H))
    for i in range(len(strong_points)):
        for t in T:
            x_idx = round(strong_points[i][0] + radius_of_coins[j]*math.cos(t))
            y_idx = round(strong_points[i][1] + radius_of_coins[j]*math.sin(t))
            if (x_idx >= 0 and x_idx < W) and (y_idx >= 0 and y_idx < H):
                votes_for_radius[x_idx, y_idx] += 1

    votes_total += votes_for_radius
    best_centers = k_largest_index_argsort(
        votes_for_radius, number_of_coins[j])
    circles[j] = best_centers
    print(f'done with {radius_of_coins[j]}')
    print(best_centers)


dispay_image(zscore(votes_total))
# print(circles)
# print(circles[0])
# print(circles[0][0])
# print(circles[0][0][0])


print('plotting final grid')
final_grid = np.zeros(np.shape(intensity_total))
for i in range(len(radius_of_coins)):
    for j in range(len(circles[i][0])):
        final_grid += draw_circle(final_grid,
                                  circles[i][0], radius_of_coins[i])
print('done')

dispay_image(zscore(final_grid))
