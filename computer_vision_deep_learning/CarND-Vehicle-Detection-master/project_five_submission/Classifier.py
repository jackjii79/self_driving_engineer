#!/usr/bin/python3.5
import Image
import numpy as np
import cv2
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time

class Classifier:
    def __init__(self, pickle_file):
        dist_pickle = pickle.load(open(pickle_file, "rb"))
        self.train_file, self.train_labels, self.test_file, self.test_labels = dist_pickle["train_x"], dist_pickle["train_y"], dist_pickle["test_x"], dist_pickle["test_y"]
        self.train_features, self.test_features, self.classifier, self.X_scaler = [], [], LinearSVC(), None
        self.pickle_clf = {}

    def generate_features(self, img, spatial_feature=True, hist_feature=True, hog_feature=True, hist_bins=32, spatial_resolution=32, pixels_per_cell=8, cells_per_block=2, color_space='LUV', hog_channel='ALL'):
        imgobj = Image.Image(img, spatial_feature, hist_feature, hog_feature, hist_bins, spatial_resolution, pixels_per_cell, cells_per_block, color_space, hog_channel)
        return imgobj.extract_features()

    def pattern_recognition_process(self, spatial_feature=True, hist_feature=True, hog_feature=True, hist_bins=32, spatial_resolution=32, pixels_per_cell=8, cells_per_block=2, color_space='LUV', hog_channel='ALL'):
        self.pickle_clf['is_spatial'], self.pickle_clf['is_hist'], self.pickle_clf['is_hog'], self.pickle_clf['num_bins'], self.pickle_clf['spatial_size'] = spatial_feature, hist_feature, hog_feature, hist_bins, spatial_resolution
        self.pickle_clf['pixels_per_cell'], self.pickle_clf['cells_per_block'], self.pickle_clf['color_space'], self.pickle_clf['hog_channel'] = pixels_per_cell, cells_per_block, color_space, hog_channel
        for fname in self.train_file:
            img = cv2.imread(fname)
            sig_feature = self.generate_features(img, spatial_feature=True, hist_feature=True, hog_feature=True, hist_bins=32, spatial_resolution=32, pixels_per_cell=8, cells_per_block=2, color_space='LUV', hog_channel='ALL')
            self.train_features.append(sig_feature)
        self.train_features = np.array(self.train_features, dtype=np.float64)

        for fname in self.test_file:
            img = cv2.imread(fname)
            sig_feature = self.generate_features(img, spatial_feature=True, hist_feature=True, hog_feature=True, hist_bins=32, spatial_resolution=32, pixels_per_cell=8, cells_per_block=2, color_space='LUV', hog_channel='ALL')
            self.test_features.append(sig_feature)
        self.test_features = np.array(self.test_features, dtype=np.float64)

        #fit a scaler
        self.X_scaler = StandardScaler().fit(self.train_features)
        #apply scaling and shifting to both train and test set
        self.train_features = self.X_scaler.transform(self.train_features)
        self.test_features = self.X_scaler.transform(self.test_features)

    def train_process(self, output_pickle):
        t1 = time.time()
        self.classifier.fit(self.train_features, self.train_labels)
        t2 = time.time()
        print(round(t2-t1, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.classifier.score(self.test_features, self.test_labels), 4))
        self.pickle_clf['svc'] = self.classifier
        self.pickle_clf['scaler'] = self.X_scaler
        pickle.dump(self.pickle_clf, open(output_pickle, "wb"))
