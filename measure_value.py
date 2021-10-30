import numpy as np
import cv2


def dispay_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(img, down_points):
    return cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)


img = cv2.imread('19.2 capstone_coins.png')
img_resized = resize_image(img, (500, 500))

dispay_image(img_resized)

print(type(img_resized))
