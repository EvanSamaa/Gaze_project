
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import AversionSignalDrivenSaliency
from Geometry_Util import rotation_angles_frome_positions
import numpy as np
from matplotlib import pyplot as plt
class Scavenger_based_planner:
    def __init__(self, saliency_maps, scene_info):
        # hyper-parameters
        self.smoothing_constant = 0.2
        self.kappa = 1 # this is the distance factor (i.e. cost of migration)
        self.phi = 1 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 10 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.2 # this specified how closely two nearby saccade can be with one another.
        # get the dt
        self.dt = saliency_maps[0]._dt
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
        # get the position of objects
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_frome_positions(self.object_positions) / 180 * np.pi
        # ========================================= state variables ==========================================
        self.current_look_at = 0
    def compute(self, initial_target):
        # variable to store the output values
        output_t = []
        output_target = []
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
        values = np.zeros(self.saliency_maps_arrs[0].shape)
        for map_arr in self.saliency_maps_arrs:
            values += map_arr
        # initialize the first look at point with the user speficied initial target
        current_look_at = initial_target
        time_within_patch = 0
        # add the first target to the output list
        output_target.append(current_look_at)
        output_t.append(0)
        temp_arr = 0
        for i in range(0, self.saliency_maps_arrs[0].shape[0]):
            ##############################################################################################
            ############################### decide whether to switch patch ###############################
            ##############################################################################################
            # compute rho (value)
            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
            look_at_mask[current_look_at] = 1
            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
            # compute the distance to the patch (first use this variable to store the position of look_at_point)
            distance_to_patch = np.tile(self.object_positions[current_look_at:current_look_at+1], [self.object_positions.shape[0], 1])
            distance_to_patch = (distance_to_patch - self.object_positions)
            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
            # compute distance-weighted patch value rho
            rho = values[i] * np.exp(-self.kappa * distance_to_patch)
            # compute Q, the expected return of leaving the current patch and move to another patch
            Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)


            # compute g_patch, the instetaneous gain by staying at the current patch
            if rho[current_look_at] > 0:
                g_patch = rho[current_look_at] * np.exp(-self.phi / values[i, current_look_at] * time_within_patch)
            else:
                g_patch = 0
            # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
            p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
            # if g_patch > Q:
            #     p_stay = 1
            # else:
            #     p_stay = 0

            ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
            # if the sampling determine that there is a patch switch
            if p_stay <= 0.7:
                rv = np.random.binomial(1, p_stay)
            else:
                rv = 1
            if rv == 0 and time_within_patch >= self.min_saccade_time:
                # TODO: make the new patch randomly sampled instead of deterministic
                new_patch = np.argmax(rho * not_looked_at_mask)
                time_within_patch = 0
                current_look_at = new_patch
                output_target.append(current_look_at)
                output_t.append(self.dt * i)
            else:
                time_within_patch += self.dt
        return output_t, output_target


