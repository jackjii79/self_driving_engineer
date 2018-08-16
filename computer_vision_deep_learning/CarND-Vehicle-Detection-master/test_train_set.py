#!/usr/bin/python3.5
import glob
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

global_vehicle_dir = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/vehicles/vehicles/"
global_non_vehicle_dir = "/home/peterhan/coding_exercise/udacity/self_driving_engineer/computer_vision_deep_learning/CarND-Vehicle-Detection-master/non-vehicles/non-vehicles/"
vehicle_train_dir, vehicle_test_dir = [global_vehicle_dir+"GTI_Far_train/*.png", global_vehicle_dir+"GTI_Left_train/*.png", global_vehicle_dir+"GTI_MiddleClose_train/*.png", global_vehicle_dir+"GTI_Right_train/*.png"], [global_vehicle_dir+"GTI_Far_test/*.png", global_vehicle_dir+"GTI_Left_test/*.png", global_vehicle_dir+"GTI_MiddleClose_test/*.png", global_vehicle_dir+"GTI_Right_test/*.png"]
full_vehicle_dir = global_vehicle_dir + "KITTI_extracted/*.png"
notvehicle_dir = [global_non_vehicle_dir+"Extras/*.png", global_non_vehicle_dir+"GTI/*.png"]

cars_train, cars_test = [], []
for traindir, testdir in zip(vehicle_train_dir, vehicle_test_dir):
    ftrains, ftests = glob.glob(traindir), glob.glob(testdir)
    cars_train.extend(ftrains)
    cars_test.extend(ftests)
x_cars = glob.glob(full_vehicle_dir)
y_cars = np.ones(len(x_cars))
x_trains, x_tests, y_trains, y_tests = train_test_split(x_cars, y_cars, test_size=0.25, random_state=42)
cars_train.extend(x_trains)
cars_test.extend(x_tests)
num_cars_train = len(cars_train)
num_cars_test = len(cars_test)
print(num_cars_train,num_cars_test)

x_not_cars = []
for filedir in notvehicle_dir:
    x_not_cars.extend(glob.glob(filedir))
y_not_cars = np.zeros(len(x_not_cars))
x_trains, x_tests, y_trains, y_tests = train_test_split(x_not_cars, y_not_cars, test_size=0.25, random_state=84) 
not_cars_train = x_trains
not_cars_test = x_tests
num_not_cars_train = len(not_cars_train)
num_not_cars_test = len(not_cars_test)
print(num_not_cars_train,num_not_cars_test)
final_train_x, final_train_y = np.concatenate((cars_train,not_cars_train)), np.concatenate((np.ones(num_cars_train),np.zeros(num_not_cars_train)))
final_test_x, final_test_y = np.concatenate((cars_test,not_cars_test)), np.concatenate((np.ones(num_cars_test),np.zeros(num_not_cars_test)))
final_train_x, final_train_y = shuffle(final_train_x, final_train_y, random_state=40)
final_test_x, final_test_y = shuffle(final_test_x, final_test_y, random_state=10)
disk_pickle = {}
disk_pickle["train_x"] = final_train_x
disk_pickle["train_y"] = final_train_y
disk_pickle["test_x"] = final_test_x
disk_pickle["test_y"] = final_test_y
pickle.dump(disk_pickle, open("train_test_pickle.p", "wb"))
