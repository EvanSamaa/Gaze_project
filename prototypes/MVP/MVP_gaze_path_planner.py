from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import AversionSignalDrivenSaliency
from Geometry_Util import rotation_angles_frome_positions
from Signal_processing_utils import dx_dt
import numpy as np
from matplotlib import pyplot as plt
import random
class Responsive_planner_simple:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, self_id=-1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.directgaze_comfy_threshold = 3 # need to look away after 6 seconds.         
        self.aversion_comfy_threshold = 2 # need to look back at the parter after 6 seconds
        self.min_eye_contact_threshold = 3 # how short should an average eye contact be 
        
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 3.0
        self.phi = 1 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.4 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.1 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info = scene
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size to padd add the salience map
        self.self_id = self_id
        object_count = 0
        time_count = 0
        max_id = 0
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] > object_count:
                object_count = arr.shape[1]
                max_id = i
            if arr.shape[0] > time_count:
                time_count = arr.shape[0]
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] < object_count:
                extension = np.zeros((arr.shape[0], object_count-arr.shape[1]))
                arr = np.concatenate([arr, extension], axis=1)
            if arr.shape[0] < time_count:
                extension = np.zeros((time_count - arr.shape[0], object_count))
                arr = np.concatenate([arr, extension], axis=0)
            self.saliency_maps_arrs.append(arr)
        self.saliency_maps_arrs = np.array(self.saliency_maps_arrs)
        # get the position of objects
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_frome_positions(self.object_positions) / 180 * np.pi
        # get the conversation partner's id
        self.conversation_partner_id = self.scene_info.object_pos.shape[0]
    def compute(self):
        # sum up the difference salience maps 
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values))
        d_val = d_val.sum(axis=1) 
        # compute the look-at-points
        output_target = [self.conversation_partner_id]
        output_t = [0]
        idx = 0
        # normalized_object_positions = self.object_positions / np.linalg.norm(self.object_positions, axis=1, keepdims=True) 
        normalized_object_positions =  self.object_positions
        while idx < self.values.shape[0]-1:
            t = idx* self.dt
            current_target = output_target[-1]
            highest_salience = np.argmax(self.values[idx+1])
            prev_t = output_t[-1] # previous gaze shift happened at prev_t      
            # if there is a change in the salience list       
            if d_val[idx] >= 1:
                # otherwise, if we have been looking away, and need to look at the
                # conversation partner, we look at the partner. 
                if (highest_salience == self.conversation_partner_id and 
                current_target != self.conversation_partner_id):
                    if t - prev_t >= 0.4: # we have to make sure the motion is not too rapid
                        output_target.append(highest_salience)
                        output_t.append(t)
                elif current_target != self.conversation_partner_id and t - prev_t >= self.aversion_comfy_threshold:
                    output_target.append(self.conversation_partner_id)
                    output_t.append(t)
                elif current_target == self.conversation_partner_id and t - prev_t >= self.min_eye_contact_threshold:  
                    time_within_patch = t - output_t[-1]
                    ##############################################################################################
                    ############################### decide whether to switch patch ###############################
                    ##############################################################################################
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                    # compute the distance to the patch (first use this variable to store the position of look_at_point)
                    distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                    distance_to_patch = (distance_to_patch - normalized_object_positions)
                    vertical_distance = (normalized_object_positions - distance_to_patch)[:, 1]
                    vertical_bias_factor = np.exp(vertical_distance * (self.top_bias - 0.5)) # if the direction is the same then the factor will be big, otherwise small 
                    
                    distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                    # compute distance-weighted patch value rho 
                    rho = self.values[i] * np.exp(-self.kappa * distance_to_patch) * vertical_bias_factor
                    # compute Q, the expected return of leaving the current patch and move to another patch
                    Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    # compute g_patch, the instetaneous gain by staying at the current patch
                    if rho[current_target] > 0:
                        g_patch = rho[current_target] * np.exp(-self.phi / self.values[i, current_target] * time_within_patch)
                    else:
                        g_patch = 0
                    # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                    p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                    ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                    # if the sampling determine that there is a patch switch
                    if p_stay <= 0.5:
                        rv = np.random.binomial(1, p_stay)
                    else:
                        rv = 1
                    if rv == 0 and time_within_patch >= self.min_saccade_time:
                        # TODO: make the new patch randomly sampled instead of deterministic
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(self.dt * idx)
            elif current_target != self.conversation_partner_id and highest_salience == self.conversation_partner_id and t - prev_t >= self.aversion_comfy_threshold:
                if t - prev_t >= 0.4: # we have to make sure the motion is not too rapid
                    output_target.append(highest_salience)
                    output_t.append(t)
            idx = idx + 1
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target