#!/usr/bin/python3.5
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Image:

    def __init__(self, undist_img):
        self.image = undist_img
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.uninitialization = True

    def locate_roi(self):
        results = np.empty((4,2), np.float32)
        results[0] = [390+100, 102+380]
        results[1] = [710+100, 102+380]
        results[2] = [self.image.shape[1], self.image.shape[0]]
        results[3] = [0, self.image.shape[0]]
        return results

    def order_points(self, pts):
        rect = np.zeros((4,2), np.float32)
        pts = pts[pts[:,1].argsort()]
        if pts[0][0] < pts[1][0]:
            rect[0] = pts[0] #top left
            rect[1] = pts[1] #top right
        else:
            rect[0] = pts[1]
            rect[1] = pts[0]
        if pts[2][0] < pts[3][0]:
            rect[2] = pts[3] #bottom right
            rect[3] = pts[2] #bottom left
        else:
            rect[2] = pts[2]
            rect[3] = pts[3]
        return rect

    def perspective_transform(self, input_img):
        pts = self.locate_roi()
        rect = self.order_points(pts)
        img_size = (input_img.shape[1], input_img.shape[0])

        #reversed order (col, row) -- (width, height)
        dst = np.array([[0,0], [img_size[0],0], [img_size[0],img_size[1]], [0,img_size[1]]], np.float32)
        self.M = cv2.getPerspectiveTransform(rect, dst)
        self.MINV = cv2.getPerspectiveTransform(dst, rect)
        self.warped_image = cv2.warpPerspective(input_img, self.M, img_size, flags=cv2.INTER_LINEAR)
        return self.warped_image, self.MINV

    def threshold_mask(self, flag, thres_min, thres_max, sobel_kernel=5):
        if flag == 'x' or flag == 'y':
            binary_output = self.absolute_sobel_thresh(sobel_kernel, flag, thres_min, thres_max)
        elif flag == 'mag':
            binary_output = self.magnitude_sobel_thresh(sobel_kernel, thres_min, thres_max)
        elif flag == 'dir':
            binary_output = self.direction_sobel_thresh(sobel_kernel, thres_min, thres_max)
        else:
            binary_output = self.color_thresh(flag, thres_min, thres_max)

        if self.uninitialization == True:
            self.filter = np.zeros_like(binary_output)
            self.filter[(binary_output == 1)] = 1
            self.uninitialization = False

        else:
            new_filter = np.zeros_like(self.filter)
            new_filter[(binary_output == 1) & (self.filter == 1)] = 1
            self.filter = new_filter

    def absolute_sobel_thresh(self, sobel_kernel, orient, thresh_min, thresh_max):
        if orient == 'x':
            sobel_gradient = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel_gradient = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_gradient = np.absolute(sobel_gradient)
        sobel_gradient = np.uint8(255*sobel_gradient/np.max(sobel_gradient))
        binary_output = np.zeros_like(sobel_gradient)
        binary_output[(sobel_gradient >= thresh_min) & (sobel_gradient <= thresh_max)] = 1
        return binary_output

    def magnitude_sobel_thresh(self, sobel_kernel, thresh_min, thresh_max):
        sobel_x, sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel), cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        sobel_magnitude = np.uint8(255*sobel_magnitude/np.max(sobel_magnitude))
        binary_output = np.zeros_like(sobel_magnitude)
        binary_output[(sobel_magnitude >= thresh_min) & (sobel_magnitude <= thresh_max)] = 1
        return binary_output

    def direction_sobel_thresh(self, sobel_kernel, thresh_min, thresh_max):
        sobel_x, sobel_y = np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)), np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        sobel_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        binary_output = np.zeros_like(sobel_direction)
        binary_output[(sobel_direction >= thresh_min) & (sobel_direction <= thresh_max)] = 1
        return binary_output

    def color_thresh(self, channel, thresh_min, thresh_max):
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        if channel == 's':
            channel_val = hls[:,:,2]
        elif channel == 'l':
            channel_val = hls[:,:,1]
        else:
            channel_val = hls[:,:,0]
        binary_output = np.zeros_like(channel_val)
        binary_output[(channel_val >= thresh_min) & (channel_val <= thresh_max)] = 1
        return binary_output

    def get_final_filter(self):
        return self.filter

    def draw_images(self, im2, string):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(self.image)
        ax1.set_title('original image', fontsize=30)
        ax2.imshow(im2, cmap='gray')
        ax2.set_title(string, fontsize=30)
        plt.subplots_adjust(left=0,right=1,top=.9,bottom=0.)
        plt.show()
