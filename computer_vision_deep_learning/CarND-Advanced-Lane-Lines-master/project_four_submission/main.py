#!/usr/bin/python3.5
import Lane
import Line
import Image
import pickle
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip

pick_file = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/wide_dist_pickle.p"
sobel_min_x, sobel_max_x = 5, 150
sobel_min_y, sobel_max_y = 3, 150
mag_min, mag_max = 10, 150
dir_min, dir_max = -60, 60
col_min_s, col_max_s = 100, 255
#80
#15
col_min_l, col_max_l = 100, 255
#100
#100
sobel_kernel = 5
cache_size = 10
nwindows, minpixel, margin = 9, 50, 100
dist_pickle = pickle.load(open(pick_file, "rb"))
mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]
lines = Line.Line(cache_size)
output_video_file = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/output3.mp4"

def full_pipeline(input_image):
    new_image = cv2.undistort(input_image, mtx, dist, None, mtx)
    img_obj = Image.Image(new_image.copy())
    img_obj.threshold_mask('mag', mag_min, mag_max, sobel_kernel)
    img_obj.threshold_mask('dir', dir_min, dir_max, sobel_kernel)
    img_obj.threshold_mask('s', col_min_s, col_max_s)
    img_obj.threshold_mask('l', col_min_l, col_max_l)
    warped, inv_M = img_obj.perspective_transform(img_obj.get_final_filter().copy())
    warped_img, inv_M2 = img_obj.perspective_transform(new_image.copy())
    lane_obj = Lane.Lane(warped)
    if lines.detected == True:
        is_detect_left, is_detect_right, leftfit, rightfit = lane_obj.get_lane_fitting(True, lines.best_left_fit, lines.best_right_fit)
    else:
        is_detect_left, is_detect_right, leftfit, rightfit = lane_obj.get_lane_fitting(False)
    lane_obj.compute_curvature()
    lane_obj.compute_horizontal_distance()
    lane_obj.compute_center_offset()
    lines.verify_last_fit(lane_obj.detect_lane_left, lane_obj.detect_lane_right, lane_obj.leftfit_pixel, lane_obj.rightfit_pixel, lane_obj.h_dist)
    result_img = lane_obj.draw_fitting_lane(input_image, inv_M, lines.best_left_fit, lines.best_right_fit)
    return result_img

'''
clip1 = VideoFileClip("/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/challenge_video.mp4")
white_clip = clip1.fl_image(full_pipeline)
white_clip.write_videofile(output_video_file, audio=False)
'''
imgpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/test_images/*.jpg"
outpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/output_image"
fnames = glob.glob(imgpath)
for idx,fname in enumerate(fnames):
    input_img = mpimg.imread(fname)
    resultimg = full_pipeline(input_img)
    fig = plt.figure()
    plt.imshow(resultimg)
    fig.savefig(outpath+str(idx+1)+".jpg")
'''
imgpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/images_test/images*.jpg"
outpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/images_output/images"
fnames = glob.glob(imgpath)
idx = 0
for fname in fnames:
    input_img = mpimg.imread(fname)
    result_img, res_img = full_pipeline(input_img)
    fig = plt.figure()
    plt.imshow(result_img, cmap='gray')
    fig.savefig(outpath+str(idx)+".jpg")
    fig = plt.figure()
    plt.imshow(res_img)
    fig.savefig(outpath+str(idx+1)+".jpg")
    idx += 2

vertical_flip = lambda frame:frame
clip1 = VideoFileClip("/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/harder_challenge_video.mp4")
imgpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/images_test/images%03d.jpg"
clip1.fl_image(vertical_flip).to_images_sequence(imgpath)
'''
