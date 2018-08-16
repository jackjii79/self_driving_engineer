#!/usr/bin/python3.5
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def Camera_calibrate(imgfiles):
    objps = np.zeros((9*6,3), np.float32)
    objps[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints, imgpoints = [], []
    fnames = glob.glob(imgfiles)
    for idx, imgfile in enumerate(fnames):
        img = cv2.imread(imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            objpoints.append(objps)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (gray.shape[1],gray.shape[0]), None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("wide_dist_pickle.p", "wb"))
    return mtx, dist

def Generate_Undistort_images(imgfiles):
    mtx, dist = Camera_calibrate(imgfiles)
    fnames = glob.glob(imgfiles)
    for index, fname in enumerate(fnames):
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original_image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted image', fontsize=30)
        plt.show()
    return

imgfiles = '/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/camera_cal/calibration*.jpg'
Generate_Undistort_images(imgfiles)
