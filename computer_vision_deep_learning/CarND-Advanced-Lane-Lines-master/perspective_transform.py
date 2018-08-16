#!/usr/bin/python3.5
import numpy as np
import cv2
import matplotlib.image as mpg
import matplotlib.pyplot as plt
import pickle

#pts:(row, col) -- (height, wdith)
def order_points(pts):
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

def four_points_transform(image, pts):
    rect = order_points(pts)
    img_size = (image.shape[1], image.shape[0])

    #reversed order (col, row) -- (width, height)
    dst = np.array([[0,0], [img_size[0],0], [img_size[0],img_size[1]], [0,img_size[1]]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    MINV = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, MINV

def locate_four_points(image):
    results = np.empty((4,2), np.float32)
    results[0] = [390+100, 102+380]
    results[1] = [710+100, 102+380]
    results[2] = [image.shape[1], image.shape[0]]
    results[3] = [0, image.shape[0]]
    return results

'''
imgfile = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/test_images/test6.jpg"
picklefile = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Advanced-Lane-Lines-master/wide_dist_pickle.p"
dist_pickle = pickle.load(open(picklefile, "rb"))
mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]
#src = np.array([[490,482],[810,482],[1250,720],[40,720]], np.float32)
#pts1, pts2, pts3, pts4 = [490,482], [810,482], [1250,720], [40,720]
#src = np.array([pts3,pts1,pts4,pts2], np.float32)
image = mpg.imread(imgfile)
image = cv2.undistort(image, mtx, dist, None, mtx)
#image = image[380:-30,100:-100,:]
src = locate_four_points(image)
cv2.circle(image, tuple(src[0]), 1, [255,0,0], 10)
cv2.circle(image, tuple(src[1]), 1, [255,0,0], 10)
cv2.circle(image, tuple(src[2]), 1, [255,0,0], 10)
cv2.circle(image, tuple(src[3]), 1, [255,0,0], 10)
warp = four_points_transform(image, src)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.tight_layout()
ax1.imshow(image)
ax2.imshow(warp)
plt.subplots_adjust(left=0,right=1,top=.9,bottom=0.)
plt.show()
'''
