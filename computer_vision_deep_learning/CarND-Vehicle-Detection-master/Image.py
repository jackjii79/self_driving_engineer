#!/usr/bin/python3.5
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

class Image:
    def __init__(self, img, spatial_feature=True, hist_feature=True, hog_feature=True, hist_bins=32, spatial_resolution=32, pixels_per_cell=8, cells_per_block=2, color_space='LUV', hog_channel='ALL'):
        self.img, self.is_spatial, self.is_hist, self.is_hog, self.num_bins, self.spatial_size, self.cell_size, self.block_size, self.color_space, self.hog_channel = img, spatial_feature, hist_feature, hog_feature, hist_bins, spatial_resolution, pixels_per_cell, cells_per_block, color_space, hog_channel
        self.features_list = []

    def spatial_bin_features(self, img):
        spatial_feature = cv2.resize(img, (self.spatial_size, self.spatial_size)) 
        self.features_list.append(spatial_feature)
        return spatial_feature.ravel()

    def color_hist_features(self, img):
        channel1_hist = np.histogram(img[:,:,0], bins=self.num_bins, range=(0,256))
        channel2_hist = np.histogram(img[:,:,1], bins=self.num_bins, range=(0,256))
        channel3_hist = np.histogram(img[:,:,2], bins=self.num_bins, range=(0,256))
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def hog_features(self, img, feature_vec=True):
        hog_feature, hog_img = hog(img, orientations=9, pixels_per_cell=(self.cell_size,self.cell_size), cells_per_block=(self.block_size,self.block_size), block_norm='L2-Hys', transform_sqrt=True, visualise=True, feature_vector=feature_vec)
        self.features_list.append(hog_img)

        return hog_feature, hog_img

    def extract_features(self):
        features = []
        if self.color_space != 'BGR':
            if self.color_space == 'HSV':
                self.feature_img = cv2.cvtColor(self.img, cv2.COLOR_BGRB2HSV)
            elif self.color_space == 'LUV':
                self.feature_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                self.feature_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                self.feature_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                self.feature_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        else:
            self.feature_img = self.img.copy()

        if self.is_spatial == True:
            features.append(self.spatial_bin_features(self.feature_img))

        if self.is_hist == True:
            features.append(self.color_hist_features(self.feature_img))

        if self.is_hog == True:
            if self.hog_channel == 'ALL':
                hog_feature = []
                for channel in range(self.feature_img.shape[2]):
                    hog_feature.append(self.hog_features(self.feature_img[:,:,channel]))
                hog_feature = np.ravel(hog_feature)
            else:
                hog_feature = self.hog_features(self.feature_img[:,:,self.hog_channel])
            features.append(hog_feature)
        return np.concatenate(features)

    def draw_features(self, feature, str_input):
        plt.title(str_input)
        plt.imshow(feature, cmap='gray')
        plt.show()

    def draw_histogram(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        ax1.set_title('channel 1 histogram')
        ax1.hist(self.feature_img[:,:,0], bins=self.num_bins, range=(0,256))
        ax1.set_title('channel 2 histogram')
        ax1.hist(self.feature_img[:,:,1], bins=self.num_bins, range=(0,256))
        ax1.set_title('channel 3 histogram')
        ax1.hist(self.feature_img[:,:,2], bins=self.num_bins, range=(0,256))
        plt.show()
