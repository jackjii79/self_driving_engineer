#!/usr/bin/python3.5
import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.measurements import label

class Detection:
    def __init__(self, img, pickfile):
        dist_pickle = pickle.load(open(pickfile, "rb"))
        self.clf, self.X_scaler, self.is_spatial, self.is_hist, self.is_hog, self.num_bins, self.spatial_size, self.pix_per_cell, self.cell_per_block, self.color_space, self.hog_channel = \
        dist_pickle['svc'], dist_pickle['scaler'], dist_pickle['is_spatial'], dist_pickle['is_hist'], dist_pickle['is_hog'], dist_pickle['num_bins'], dist_pickle['spatial_size'], \
        dist_pickle['pixels_per_cell'], dist_pickle['cells_per_block'], dist_pickle['color_space'], dist_pickle['hog_channel']
        self.img, self.imgobj = img, Image.Image(img, self.is_spatial, self.is_hist, self.is_hog, self.num_bins, self.spatial_size, self.pix_per_cell, self.cell_per_block, self.color_space, self.hog_channel)
        self.boxing_list, self.heatmap = [], np.zeros_like(self.img)

    def search_windows(self, scale, y_start, y_end, x_start):
        draw_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        input_img = self.img[y_start:y_end,x_start:,:]
        
        if self.color_space != 'BGR' or self.color_space != 'RGB':
            if self.color_space == 'HSV':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRB2HSV)
            elif self.color_space == 'LUV':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
        
        if scale != 1:
            input_img = cv2.resize(input_img, (np.int(input_img.shape[1]/scale), np.int(input_img.shape[0]/scale)))
        
        channel_1 = input_img[:,:,0]
        channel_2 = input_img[:,:,1]
        channel_3 = input_img[:,:,2]
        print(channel_1.shape)

        #extract the whole hog feature once and for all
        hog_feature_1, hog1 = self.imgobj.hog_features(channel_1, feature_vec=False)
        hog_feature_2, hog2 = self.imgobj.hog_features(channel_2, feature_vec=False)
        hog_feature_3, hog3 = self.imgobj.hog_features(channel_3, feature_vec=False)
        
        #num of possible positions for block in x-axis and y-axis
        num_blocks_x = (channel_1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        num_blocks_y = (channel_1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1

        #64 pixels has to be 64 x 64 since training data is based on 64x64 !!!
        window = 64

        #num of possible positions for block in window size
        num_blocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2

        #total number of steps for shifting window in x-axis and y-axis, note if without dividing by cells_per_step, then cells_per_step is just 1 !!
        num_steps_x = (num_blocks_x - num_blocks_per_window) // cells_per_step + 1
        num_steps_y = (num_blocks_y - num_blocks_per_window) // cells_per_step + 1

        for xs in range(num_steps_x):
            for ys in range(num_steps_y):
                x_s, x_e = xs * cells_per_step, xs * cells_per_step + num_blocks_per_window
                y_s, y_e = ys * cells_per_step, ys * cells_per_step + num_blocks_per_window

                if self.hog_channel == 'ALL':
                    hog_feature1 = hog_feature_1[y_s:y_e, x_s:x_e].ravel()
                    hog_feature2 = hog_feature_2[y_s:y_e, x_s:x_e].ravel()
                    hog_feature3 = hog_feature_3[y_s:y_e, x_s:x_e].ravel()
                    hog_features = np.hstack((hog_feature1, hog_feature2, hog_feature3))

                elif self.hog_channel == 0:
                    hog_features = hog_feature_1[y_s:y_e, x_s:x_e].ravel()

                elif self.hog_channel == 1:
                    hog_features = hog_feature_2[y_s:y_e, x_s:x_e].ravel()

                else:
                    hog_features = hog_feature_3[y_s:y_e, x_s:x_e].ravel()

                x_top_left, y_top_left = x_s * self.pix_per_cell, y_s * self.pix_per_cell
                subimg = cv2.resize(input_img[y_top_left:y_top_left+window, x_top_left:x_top_left+window], (window,window))

                spatial_features = self.imgobj.spatial_bin_features(subimg)
                hist_features = self.imgobj.color_hist_features(subimg)

                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
                test_prediction = self.clf.predict(test_features)

            if test_prediction == 1:
                x_left = np.int(x_top_left * scale)
                y_left = np.int(y_top_left * scale)
                wins = np.int(window * scale)
                self.boxing_list.append(((x_left+x_start, y_left+y_start),(x_left+x_start+wins, y_left+y_start+wins)))
                cv2.rectangle(draw_img, (x_left+x_start, y_left+y_start), (x_left+x_start+wins, y_left+y_start+wins), (0,0,255), 6)

        return draw_img

    def multiple_scale_window(self, scales, y_start_stops):
        length = len(scales)
        for scale, y_range in zip(scales, y_start_stops):
            #tile_str = "scale is " + str(scale) + ", y_start is " + str(y_range[0]) + " and y_end is " + str(y_range[1]) + " and x_start is " + str(y_range[2])
            self.search_windows(scale, y_range[0], y_range[1], y_range[2])
            #plt.imshow(self.search_windows(scale, y_range[0], y_range[1], y_range[2]))
            #plt.title(tile_str)
            #plt.show()
        return

    def process_heatmap(self, threshold):
        for box in self.boxing_list:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1   
        result = 0
        if (self.heatmap > 1).any() == True:
            result = np.int(np.mean(self.heatmap[self.heatmap > 1]))
        self.heatmap[self.heatmap <= threshold] = 0
        self.heatmap = label(self.heatmap)
        return result

    def draw_heatmap(self):
        print("there are ",self.heatmap[1]," cars founded")
        plt.imshow(np.array(self.heatmap[0],dtype=np.float32), cmap='hot')
        plt.title('heatmap')
        plt.show()

    def draw_label_box(self):
        draw_img = self.img.copy()
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        for idx in range(1, self.heatmap[1]+1):
            nonzero = (self.heatmap[0] == idx).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox),np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
        #plt.imshow(draw_img)
        #plt.title('final output')
        #plt.show()
        return draw_img
