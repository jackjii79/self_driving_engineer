#!/usr/bin/python3.5
from moviepy.editor import VideoFileClip

vertical_flip = lambda frame:frame
clip1 = VideoFileClip("/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/project_video.mp4")
imgpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/images_test/images%03d.jpg"
clip1.fl_image(vertical_flip).to_images_sequence(imgpath)
