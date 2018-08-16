#!/usr/bin/python3.5
import Classifier
import Detection
import cv2
import glob
from moviepy.editor import VideoFileClip
import numpy as np

scales = [3,2.5,2,1.5,1.25,1]
y_start_stops = [[465,660,550],[430,600,550],[400,540,550],[380,500,550],[390,490,550],[400,483,550]]
picklefile = "train_model0.p"
videopath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/project_video.mp4"
outputpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/project_output.mp4"
#videopath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/test_video.mp4"
#outputpath = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/test_output.mp4"
outdir = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/test_dir/images%03d.jpg"
vertical_flip = lambda frame:frame

class Record:
    history_track = []
    cache_size = 15

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mydetect = Detection.Detection(img, picklefile)
    mydetect.multiple_scale_window(scales, y_start_stops)
    #print(Record.history_track)
    if len(Record.history_track) > 0:
        last_threshold = mydetect.process_heatmap(np.int(np.min(np.array(Record.history_track))))
    else:
        last_threshold = mydetect.process_heatmap(2)
    if last_threshold > 0:
        Record.history_track.append(last_threshold)
    else:
        Record.history_track = []
    if len(Record.history_track) > Record.cache_size:
        Record.history_track = Record.history_track[1:]
    return mydetect.draw_label_box()

#clip = VideoFileClip(videopath).subclip(20,)
clip = VideoFileClip(videopath)
white_clip = clip.fl_image(process_image)
white_clip.write_videofile(outputpath, audio=False)
#clip.fl_image(vertical_flip).to_images_sequence(outdir)

'''
output_file = "train_model0.p"
is_spatial = True
is_hist = True
is_hog = True
num_bins = 32
spatial_size = 32
pixels_per_cell = 8
cells_per_block = 2
color_space = 'LUV'
hog_channel = 'ALL'
pickle_file = "train_test_pickle.p"
myclf = Classifier.Classifier(pickle_file)
myclf.pattern_recognition_process(is_spatial, is_hist, is_hog, num_bins, spatial_size, pixels_per_cell, cells_per_block, color_space, hog_channel)
myclf.train_process(output_file)
'''

'''
threshold = 2
fnames = glob.glob("test_images/*.jpg")
for fname in fnames:
    print(fname)
    testimg = cv2.imread(fname)
    mydetect = Detection.Detection(testimg, picklefile)
    mydetect.multiple_scale_window(scales, y_start_stops)
    mydetect.process_heatmap(threshold)
    mydetect.draw_heatmap()
    mydetect.draw_label_box()
'''
