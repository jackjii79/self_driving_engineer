#!/usr/bin/python3.5
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpg

class Lane:

    def __init__(self, image, window_size=9, search_margin=100, pix_thresh=50):
        self.image, self.window_size, self.margin, self.pixsize = image, window_size, search_margin, pix_thresh
        self.ym_per_pixel, self.xm_per_pixel = 30/720, 3.7/700
        self.ploty = np.linspace(0, self.image.shape[0]-1, self.image.shape[0])
        self.detect_lane_left, self.detect_lane_right = False, False
        self.leftfit_pixel, self.rightfit_pixel, self.leftfit_meter, self.rightfit_meter = None, None, None, None
        self.curvature, self.h_dist, self.offset_center = None, None, None

    def compute_fitting(self):
        if self.detect_lane_left == True:
            leftx1, lefty1 = self.leftx, self.lefty
            leftx2, lefty2 = self.leftx * self.xm_per_pixel, self.lefty * self.ym_per_pixel
            left_fit1 = np.polyfit(lefty1, leftx1, 2)
            left_fit2 = np.polyfit(lefty2, leftx2, 2)
            self.leftfit_meter, self.leftfit_pixel = left_fit2, left_fit1

        if self.detect_lane_right == True:
            rightx1, righty1 = self.rightx, self.righty
            rightx2, righty2 = self.rightx * self.xm_per_pixel, self.righty * self.ym_per_pixel
            right_fit1 = np.polyfit(righty1, rightx1, 2)
            right_fit2 = np.polyfit(righty2, rightx2, 2)
            self.rightfit_meter, self.rightfit_pixel = right_fit2, right_fit1

    def search_lane(self):
        histogram = np.sum(self.image[self.image.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        leftbase, rightbase = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
        win_height = np.int(self.image.shape[0]//self.window_size)
        nonzero = self.image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        left_center, right_center = leftbase, rightbase
        left_lanes, right_lanes = [], []
        for level in range(self.window_size):
            win_y_low, win_y_high = int(self.image.shape[0]-(level+1)*win_height), int(self.image.shape[0]-level*win_height)
            win_x_left_low, win_x_left_high = left_center - self.margin, left_center + self.margin
            win_x_right_low, win_x_right_high = right_center - self.margin, right_center + self.margin
            left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
            right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]
            left_lanes.append(left_idxs)
            right_lanes.append(right_idxs)
            if len(left_idxs) > self.pixsize:
                left_center = np.int(np.median(nonzerox[left_idxs]))
            if len(right_idxs) > self.pixsize:
                right_center = np.int(np.median(nonzerox[right_idxs]))
        left_lanes = np.concatenate(left_lanes)
        right_lanes = np.concatenate(right_lanes)

        if len(left_lanes) > 0:
            self.detect_lane_left = True
            self.leftx, self.lefty = nonzerox[left_lanes], nonzeroy[left_lanes]
        if len(right_lanes) > 0:
            self.detect_lane_right = True
            self.rightx, self.righty = nonzerox[right_lanes], nonzeroy[right_lanes]

    def fast_search(self, last_leftfit, last_rightfit):
        nonzero = self.image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        left_lanes = ((nonzerox > (last_leftfit[0]*(nonzeroy**2) + last_leftfit[1]*nonzeroy + last_leftfit[2] - self.margin)) & (nonzerox < (last_leftfit[0]*(nonzeroy**2) + last_leftfit[1]*nonzeroy + last_leftfit[2] + self.margin)))
        right_lanes = ((nonzerox > (last_rightfit[0]*(nonzeroy**2) + last_rightfit[1]*nonzeroy + last_rightfit[2] - self.margin)) & (nonzerox < (last_rightfit[0]*(nonzeroy**2) + last_rightfit[1]*nonzeroy + last_rightfit[2] + self.margin)))

        if len(left_lanes) > 0:
            self.detect_lane_left = True
            self.leftx, self.lefty = nonzerox[left_lanes], nonzeroy[left_lanes]
        if len(right_lanes) > 0:
            self.detect_lane_right = True
            self.rightx, self.righty = nonzerox[right_lanes], nonzeroy[right_lanes]

    def compute_curvature(self, leftfit=None, rightfit=None):
        if (self.detect_lane_left == True and self.detect_lane_right == True) or (leftfit != None and rightfit != None):
            if leftfit == None:
                leftfit = self.leftfit_meter
            if rightfit == None:
                rightfit = self.rightfit_meter
            y_eval = (self.image.shape[0] - 1) * self.ym_per_pixel
            left_curv = ((1 + (2 * leftfit[0] * y_eval + leftfit[1]) ** 2) ** 1.5) / np.absolute(2 * leftfit[0])
            right_curv = ((1 + (2 * rightfit[0] * y_eval + rightfit[1]) ** 2) ** 1.5) / np.absolute(2 * rightfit[0])
            self.curvature = int((left_curv + right_curv) / 2)

    def compute_horizontal_distance(self, leftfit=None, rightfit=None):
        if (self.detect_lane_left == True and self.detect_lane_right == True) or (leftfit != None and rightfit != None):
            if leftfit == None:
                leftfit = self.leftfit_meter
            if rightfit == None:
                rightfit = self.rightfit_meter
            results = np.zeros(self.image.shape[0], np.float32)
            ploty = self.ploty * self.ym_per_pixel
            for idx in range(self.image.shape[0]):
                left_dist = ploty[idx] * leftfit[0] ** 2 + ploty[idx] * leftfit[1] + leftfit[2]
                right_dist = ploty[idx] * rightfit[0] ** 2 + ploty[idx] * rightfit[1] + rightfit[2]
                results[idx] = right_dist - left_dist
            self.h_dist = np.median(results)

    # > 0 means left off the center; < 0 means right off the center
    def compute_center_offset(self, leftfit=None):
        if (self.detect_lane_left == True and self.detect_lane_right == True) or leftfit != None:
            if leftfit == None:
                leftfit = self.leftfit_meter
            ploty = self.ploty * self.ym_per_pixel
            left_offs = np.zeros(self.image.shape[0], np.float32)
            for idx in range(self.image.shape[0]):
                left_offs[idx] = ploty[idx] * leftfit[0] ** 2 + ploty[idx] * leftfit[1] + leftfit[2]
            self.offset_center = round(np.median(left_offs) + self.h_dist/2 - (self.image.shape[1] * self.xm_per_pixel) / 2, 2)

    def get_lane_fitting(self, is_detected_last_time, leftfit=None, rightfit=None):
        if is_detected_last_time == True:
            self.fast_search(leftfit, rightfit)
        else:
            self.search_lane()

        self.compute_fitting()
        return self.detect_lane_left, self.detect_lane_right, self.leftfit_pixel, self.rightfit_pixel

    def get_curvature(self):
        return self.curvature

    def get_horizontal_distance(self):
        return self.h_dist

    def get_center_offset(self):
        return self.offset_center

    def draw_polygons(self, out_img):
        leftfitx = self.leftfit_pixel[0]*self.ploty**2 + self.leftfit_pixel[1]*self.ploty + self.leftfit_pixel[2]
        rightfitx = self.rightfit_pixel[0]*self.ploty**2 + self.rightfit_pixel[1]*self.ploty + self.rightfit_pixel[2]
        left_win1_lanes = np.array([np.transpose(np.vstack([leftfitx-int(self.margin/2),self.ploty]))])
        left_win2_lanes = np.array([np.flipud(np.transpose(np.vstack([leftfitx+int(self.margin/2),self.ploty])))])
        left_lane_pts = np.hstack((left_win1_lanes,left_win2_lanes))
        right_win1_lanes = np.array([np.transpose(np.vstack([rightfitx-int(self.margin/2),self.ploty]))])
        right_win2_lanes = np.array([np.flipud(np.transpose(np.vstack([rightfitx+int(self.margin/2),self.ploty])))])
        right_lane_pts = np.hstack((right_win1_lanes,right_win2_lanes))
        cv2.fillPoly(out_img, np.int_([left_lane_pts]), (0,255,0))
        cv2.fillPoly(out_img, np.int_([right_lane_pts]), (0,255,0))
        plt.imshow(out_img)
        plt.title('fitting polygon', fontsize=30)
        plt.show()

    def draw_fitting_lane(self, outimg, M1, leftfit, rightfit):
        leftfitx = leftfit[0]*self.ploty**2 + leftfit[1]*self.ploty + leftfit[2]
        rightfitx = rightfit[0]*self.ploty**2 + rightfit[1]*self.ploty + rightfit[2]
        sentence1, org1 = "Radius of Curvature = " + str(self.curvature) + "(m)", (50,50)
        sentence2, org2 = "Vehicle is " + str(self.offset_center) + "m left of center", (50,100)
        warp_zero = np.zeros_like(self.image).astype(np.uint8)
        color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
        pts_left = np.array([np.transpose(np.vstack([leftfitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightfitx, self.ploty])))])
        pts = np.hstack((pts_left,pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
        newwarp = cv2.warpPerspective(color_warp, M1, (self.image.shape[1],self.image.shape[0]))
        result_img = cv2.addWeighted(outimg, 1, newwarp, 0.4, 0)
        cv2.putText(result_img, sentence1, org1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_img, sentence2, org2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #plt.imshow(result_img)
        #plt.title('final output', fontsize=30)
        #plt.show()
        return result_img
