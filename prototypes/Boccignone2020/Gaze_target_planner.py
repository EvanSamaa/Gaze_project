from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import *
import numpy as np

class Scavenger_based_planner:
    def __init__(self, saliency_maps):
        # hyper-parameters
        self.smoothing_constant = 0.2
        # ====================================== Storing saliency maps ======================================
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size
        object_count = 0
        max_id = 0
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] > object_count:
                object_count = arr.shape[1]
                max_id = i
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] < object_count:
                extension = np.zeros((arr.shape[0], object_count-arr.shape[1]))
                arr = np.concatenate([arr, extension], axis=1)
            self.saliency_maps_arrs.append(arr)
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # ========================================= state variables ==========================================
        self.current_look_at = 0
    def compute(self, initial_target):
        # step 1: obtain smoothed saliency map
        for i in range(0, len(self.saliency_maps_arrs)):
            map_i = self.saliency_maps_arrs[i]
            smoothed_map = np.zeros(self.saliency_maps_arrs[i].shape)
            for t in range(0, map_i.shape[0]):
                smoothed_map[t] = map_i[max(0, t-1)] * self.smoothing_constant + (1 - self.smoothing_constant) * map_i[t]
            self.saliency_maps_arrs[i] = smoothed_map
        # step 2: obtain patches through thresholding, we skip this step as we don't work with images, but a list of objects
        # step 3: perform the planning

        # perform a summation for all saliency maps to obtain the overall resources:
        total_values = np.zeros(self.saliency_maps_arrs[0].shape)
        for map_arr in self.saliency_maps_arrs:
            total_values += map_arr

        current_look_at = initial_target
        time_within_patch = 0
        for i in range(0, self.saliency_maps_arrs[0].shape[0]):
            # compute rho (value)
            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
            look_at_mask[self.current_look_at] = 1
            rho =
            rho =


        return [], []


