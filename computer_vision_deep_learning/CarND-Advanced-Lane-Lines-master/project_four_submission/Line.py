#!/usr/bin/python3.5
import Lane
import numpy as np

#can be either left lanes or right lanes
class Line: 

    def __init__(self, size): #size indicates how many of the past iterations will we stores
        self.detected = False #was line been detected successfully based on previous fit
        self.past_left_fits = [] #all successful past n iterations fitting coefficients
        self.best_left_fit = None #averaged fitting overall last n iterations
        self.past_right_fits = []
        self.best_right_fit = None
        self.past_dists = []
        self.best_dist = None
        self.cache_size = size

    #the logic is if both left and right lanes are detected and within reasonable error from previously best fit results, then this is a good fitting and we append into a list of record for future best fit purpose, setting last iteration detection as True;
    #if either left or right lanes are not detected, which means zero pixels been detected as part of lanes, this is due to last best fit mistake, we remove the most recent record, setting last iteration detection as False
    #if both left and right lanes are detected, but beyond reasonable range of error, we use the previously best_fit to plot and ignore current round results, setting last iteration detection as False

    def verify_last_fit(self, is_detect_left_lane, is_detect_right_lane, last_left_fit, last_right_fit, last_h_dist):
        if is_detect_left_lane == True and is_detect_right_lane == True:
            if len(self.past_dists) > 0:
                dist_diff = abs(self.best_dist - last_h_dist)
            if len(self.past_left_fits) > 0:
                left_coeff_diff = np.absolute(last_left_fit - self.best_left_fit)
            if len(self.past_right_fits) > 0:
                right_coeff_diff = np.absolute(last_right_fit - self.best_right_fit)
            if (len(self.past_left_fits) > 0 and (left_coeff_diff[0] > 0.001 or left_coeff_diff[1] > 1. or left_coeff_diff[2] > 100.)) \
               or (len(self.past_right_fits) > 0 and (right_coeff_diff[0] > 0.001 or right_coeff_diff[1] > 1. or right_coeff_diff[2] > 100.)) \
               or (len(self.past_dists) > 0 and dist_diff > 3):
                self.detected = False
            else:
                self.detected = True
                self.past_left_fits.append(last_left_fit)
                self.past_right_fits.append(last_right_fit)
                self.past_dists.append(last_h_dist)
                if len(self.past_left_fits) > self.cache_size:
                    self.past_left_fits = self.past_left_fits[1:]
                if len(self.past_right_fits) > self.cache_size:
                    self.past_right_fits = self.past_right_fits[1:]
                self.best_left_fit = np.mean(np.array(self.past_left_fits, np.float32), axis=0)
                self.best_right_fit = np.mean(np.array(self.past_right_fits, np.float32), axis=0)
                self.best_dist = np.mean(np.array(self.past_dists, np.float32), axis=0)

        else:
            self.detected = False
            if len(self.past_dists) > 0:
                self.past_dists.pop()
            if is_detect_left_lane == False and len(self.past_left_fits) > 0:
                self.past_left_fits.pop()   
            if is_detect_right_lane == False and len(self.past_right_fits) > 0:
                self.past_right_fits.pop()   
            if len(self.past_left_fits) > 0:
                self.best_left_fit = np.mean(np.array(self.past_left_fits, np.float32), axis=0)
            else:
                self.best_left_fit = None
            if len(self.past_right_fits) > 0:
                self.best_right_fit = np.mean(np.array(self.past_right_fits, np.float32), axis=0)
            else:
                self.best_right_fit = None
            if len(self.past_dists) > 0:
                self.best_dist = np.mean(np.array(self.past_dists, np.float32), axis=0)
            else:
                self.best_dist = None

        return
