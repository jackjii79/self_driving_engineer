#!/usr/bin/python3.5
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, sobel_kernel, orient, thresh_min, thresh_max):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelabso = np.absolute(sobel)
    sobelabso = np.uint8(255*sobelabso/np.max(sobelabso))
    binary_output = np.zeros_like(sobelabso)
    binary_output[(sobelabso >= thresh_min) & (sobelabso <= thresh_max)] = 1
    return binary_output

def mag_thresh(img, sobel_kernel, thresh_min, thresh_max):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x, sobel_y = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_abso = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    sobel_abso = np.uint8(255*sobel_abso/np.max(sobel_abso))
    binary_output = np.zeros_like(sobel_abso)
    binary_output[(sobel_abso >= thresh_min) & (sobel_abso <= thresh_max)] = 1
    return binary_output

def dir_thresh(img, sobel_kernel, thresh_min, thresh_max):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    angles = np.arctan2(sobel_y, sobel_x)
    binary_output = np.zeros_like(angles)
    binary_output[(angles >= thresh_min) & (angles <= thresh_max)] = 1
    return binary_output

image = mpimg.imread('signs_vehicles_xygrad.jpg')
binary_sobel_x = abs_sobel_thresh(image, 5, 'x', 20, 100)
binary_sobel_y = abs_sobel_thresh(image, 5, 'y', 20, 100)
binary_sobel_mag = mag_thresh(image, 5, 30, 100)
binary_sobel_dir = dir_thresh(image, 5, 0.7, 1.3)
combined_sobel = np.zeros_like(binary_sobel_dir)
combined_sobel[(binary_sobel_x == 1) & (binary_sobel_y == 1) | (binary_sobel_mag == 1) & (binary_sobel_dir == 1)] = 1
print(combined_sobel)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('original image', fontsize=50)
ax2.imshow(combined_sobel, cmap='gray')
ax2.set_title('thresholded grad', fontsize=50)
plt.subplots_adjust(left=0,right=1,top=0.9,bottom=0.)
plt.show()
