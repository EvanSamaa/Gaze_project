from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import AversionSignalDrivenSaliency
from prototypes.InputDataStructures import AgentInfo_final, AgentInfo_final_multiparty
from Geometry_Util import rotation_angles_frome_positions
from Signal_processing_utils import dx_dt
import numpy as np
from matplotlib import pyplot as plt
import random

class Responsive_planner_Differnet_Targets:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, aversion_probability, aversion_probability_other, speech_activity, beats, self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 2.5
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        self.primary_gaze_target = saliency_maps[0].gaze_target_over_time
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info:AgentInfo_final = scene
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
        # pad the saliency maps that are too short
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
        self.conversation_partner_id = self.scene_info.other_speaker_id
        self.aversion_probability = aversion_probability
        self.aversion_probability_other = aversion_probability_other
        self.speech_activity = speech_activity
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_saccade_time_consecutive
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 2:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = [[0, self.conversation_partner_id]]
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        # compute the look-at-points
        output_target = [self.scene_info.other_speaker_id]
        output_t = [0]
        normalized_object_positions = self.object_positions
        # simulation time t
        t = 0
        speech_change_counter = 0 # current position in the speech_change_boundaries array         
        gaze_target_counter = 0 # current position in the primary_gaze_target array 
        swap_target = False # whether we are switching between targets
        in_aversion = False # starts out side of gaze aversions
        no_more_gaze_swaps = False
        aversion_start = -1
        aversion_end = -1
        aversion_time = -1
        
        while t < self.aversion_probability.shape[0] * self.dt + 5:
            # if we are at a scripted target change      
            if t > self.primary_gaze_target[gaze_target_counter][0] - 0.3 and not no_more_gaze_swaps:
                if gaze_target_counter == len(self.primary_gaze_target) - 1:
                    no_more_gaze_swaps = True
                # see if the active object changes
                if self.scene_info.active_object_id != self.primary_gaze_target[gaze_target_counter][1]:
                    swap_target = True # if it does, mark the swap_target variable to True
                # change the active object ID
                self.scene_info.active_object_id = self.primary_gaze_target[gaze_target_counter][1]
                if gaze_target_counter < len(self.primary_gaze_target) - 1:
                    gaze_target_counter = gaze_target_counter + 1
                if swap_target:
                    output_target.append(self.scene_info.active_object_id)
                    output_t.append(t)
                    swap_target = False
            # the case that the neural model does not predict change
            else:
                # if it's not currently a scripted change, then see if the current active object is the other speaker
                # if it is another speaker
                current_target = output_target[-1]
                if self.scene_info.active_object_id == self.scene_info.other_speaker_id:
                    index = int(round(t/self.dt))
                    index = min(index, self.values.shape[0]-1)
                    # see if there is a change in aversion/direct gaze status
                    if d_val_neural[index] > 0.5:
                        if self.aversion_probability[index] <= 0.5:
                            # if it's direct gaze
                            in_aversion = False
                            output_target.append(self.scene_info.other_speaker_id)
                            output_t.append(t)
                        else:
                            # if it's entering aversion
                            in_aversion = True
                            # record the start and end of the aversion
                            aversion_start = t
                            for i in range(index+1, d_val_neural.shape[0]):
                                if d_val_neural[i] > 0.5:
                                    aversion_end = i * self.dt
                            aversion_time = aversion_end - aversion_start
                            # compute rho (value) to determine gaze target
                            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                            look_at_mask[current_target] = 1
                            if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                                pass
                            else:
                                look_at_mask[self.conversation_partner_id] = 1
                            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                            # compute the distance to the patch (first use this variable to store the position of look_at_point)
                            distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                            distance_to_patch = (distance_to_patch - normalized_object_positions)
                            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                            # compute distance-weighted patch value rho
                            rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                    else:
                        # ================================ NEW ================================ 
                        if current_target == self.conversation_partner_id:
                            t += self.dt
                            time_within_patch = t - output_t[-1]
                            continue
                        # ================================ NEW ================================
                        time_within_patch = t - output_t[-1]
                        # use the random walk algorithm to determine whether to switch patch
                        # compute rho (value)
                        look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                        look_at_mask[current_target] = 1
                        if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                            pass
                        else:
                            look_at_mask[self.conversation_partner_id] = 1
                        not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                        # compute the distance to the patch (first use this variable to store the position of look_at_point)
                        distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                        distance_to_patch = (distance_to_patch - normalized_object_positions)
                        distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                        # compute distance-weighted patch value rho
                        rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[current_target] > 0:
                            g_patch = rho[current_target] * np.exp(-self.phi / self.values[index, current_target] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        # if the sampling determine that there is a patch switch
                        if p_stay <= 0.2:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 1
                        # see where is the nearest next impulse
                        if rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time: # we have to make sure the motion is not too rapid
                            # TODO: make the new patch randomly sampled instead of deterministicand
                            # new_patch = np.argmax(rho * not_looked_at_mask)
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            time_within_patch = 0
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                            in_aversion = True
                    # if there isn't a deep learning predicted change
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    # if there isn't an explicit change in aversion status.
                # if the person is just looking into the blank space
                # else:
                    
                #     # if there isn't a deep learning predicted change
                #     current_target = output_target[-1]
                #     time_within_patch = t - output_t[-1]
            t += self.dt
        return output_t, output_target
        
        for i in range(0, len(speech_change_boundaries)):
            # see if the next state is aversion or direct gaze
            if speech_change_boundaries[i][1] == 0:
                # if it's direct gaze
                in_aversion = False
                output_target.append(self.conversation_partner_id)
                output_t.append(speech_change_boundaries[i][0])
            else:
                # if it's aversion
                if i == len(speech_change_boundaries)-1:
                    aversion_end = speech_change_boundaries[i][0] + 2
                else:
                    aversion_end = speech_change_boundaries[i+1][0]
                aversion_start = speech_change_boundaries[i][0]
                aversion_time = aversion_end - aversion_start
                t = aversion_start
                while t < aversion_end:
                    idx = int(t / self.dt)
                    # make sure index is not out of range
                    idx = min(idx, self.values.shape[0]-1)
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    ##############################################################################################
                    ############################### decide whether to switch patch ###############################
                    ##############################################################################################
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    if self.speech_activity[idx] == 1 and self.aversion_probability_other[idx] == 1:
                        pass
                    else:
                        look_at_mask[self.conversation_partner_id] = 1
                    not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                    # compute the distance to the patch (first use this variable to store the position of look_at_point)
                    distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                    distance_to_patch = (distance_to_patch - normalized_object_positions)
                    distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                    # compute distance-weighted patch value rho
                     
                    rho = self.values[idx] * np.exp(-self.kappa * distance_to_patch * (1 / aversion_time))
                    # compute Q, the expected return of leaving the current patch and move to another patch
                    Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    # compute g_patch, the instetaneous gain by staying at the current patch
                    if rho[current_target] > 0:
                        g_patch = rho[current_target] * np.exp(-self.phi / self.values[idx, current_target] * time_within_patch)
                    else:
                        g_patch = 0
                    # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                    p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                    ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                    # if the sampling determine that there is a patch switch
                    if p_stay <= 0.2:
                        rv = np.random.binomial(1, p_stay)
                    else:
                        rv = 1
                    # see where is the nearest next impulse
                    if ((rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time 
                         ) or in_aversion == False) and time_within_patch >= self.min_saccade_time_psysiological: # we have to make sure the motion is not too rapid
                        # TODO: make the new patch randomly sampled instead of deterministicand
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(t)
                        in_aversion = True
                    t = t + self.dt
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target
class Responsive_planner_Listener_wonders:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, aversion_probability, aversion_probability_other, speech_activity, beats, 
                 self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1, wonder_list=[], maximum_mutural_gaze_during_wonder=0.2):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 2.5
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        self.primary_gaze_target = saliency_maps[0].gaze_target_over_time
        self.wonder_list = wonder_list
        self.wondering_saccade_min_saccade_time_psysiological = 0.5
        self.maximum_mutural_gaze_during_wonder = maximum_mutural_gaze_during_wonder
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info:AgentInfo_final = scene
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
        # pad the saliency maps that are too short
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
        self.conversation_partner_id = self.scene_info.other_speaker_id
        self.aversion_probability = aversion_probability
        self.aversion_probability_other = aversion_probability_other
        self.speech_activity = speech_activity
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_saccade_time_consecutive
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 2:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        d_val_neural_other = np.abs(dx_dt(self.aversion_probability_other, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = [[0, self.conversation_partner_id]]
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        print(speech_change_boundaries)
        # compute the look-at-points
        output_target = [self.scene_info.other_speaker_id]
        output_t = [0]
        normalized_object_positions = self.object_positions
        # simulation time t
        t = 0
        speech_change_counter = 0 # current position in the speech_change_boundaries array         
        gaze_target_counter = 0 # current position in the primary_gaze_target array 
        wonder_list_counter = 0 # current position in the wonder_list array
        swap_target = False # whether we are switching between targets
        in_aversion = False # starts out side of gaze aversions
        no_more_gaze_swaps = False
        no_more_wonders = False
        aversion_start = -1
        aversion_end = -1
        aversion_time = -1
        wondering=False
        wondering_wonder_timer = 0
        wondering_forced_gaze = False
        if len(self.wonder_list) <= 0:
            no_more_wonders = True
        
        while t < self.aversion_probability.shape[0] * self.dt + 5:
            # if we are at a scripted target change 
            if t > self.primary_gaze_target[gaze_target_counter][0] - 0.3 and not no_more_gaze_swaps:
                if gaze_target_counter == len(self.primary_gaze_target) - 1:
                    no_more_gaze_swaps = True
                # see if the active object changes
                if self.scene_info.active_object_id != self.primary_gaze_target[gaze_target_counter][1]:
                    swap_target = True # if it does, mark the swap_target variable to True
                # change the active object ID
                self.scene_info.active_object_id = self.primary_gaze_target[gaze_target_counter][1]
                if gaze_target_counter < len(self.primary_gaze_target) - 1:
                    gaze_target_counter = gaze_target_counter + 1
                if swap_target:
                    output_target.append(self.scene_info.active_object_id)
                    output_t.append(t)
                    swap_target = False
            # in the case that we have a scripted wondering_time
            elif wondering or (not no_more_wonders and (t > self.wonder_list[wonder_list_counter][0] - 0.3 and t < self.wonder_list[wonder_list_counter][1])):
                # if we just entered wondering state, reset time_within_patch
                if not wondering:
                    time_within_patch = 0
                # book keeping, get out of wondering state if currently wondering but wondering have ended:
                if wondering and t > self.wonder_list[wonder_list_counter][1]:
                    wondering = False
                    wonder_list_counter += 1
                    # return gaze to other speaker if not currently looking at other speaker
                    if output_target[-1] != self.scene_info.other_speaker_id:
                        output_target.append(self.scene_info.other_speaker_id)
                        output_t.append(t)
                        continue
                else:
                    wondering = True
                # if we have reached the end of the list of scripted wondering intervals, make sure no more happens. 
                if wonder_list_counter == len(self.wonder_list) - 1:
                    no_more_wonders = True
                # here is the wondering behaviour.
                # there will be two states, in state one, we look at the other speaker, in state two, we do the random walk
                # to transition into state 1, we look at aversion timing of the partner (if they look we look)
                # to transition into state 2, we compare wonder_timer and the maximum_mutural_gaze_during_wonder
                # the default state will be state 1
                
                # I first define rules of state transition, then implement state machine to be more clear
                # state transition rules:
                # 1. if the other speaker is looking at us, we end forced gaze when duration is over
                if wondering_forced_gaze:
                    if wondering_wonder_timer >= self.maximum_mutural_gaze_during_wonder:
                        # return gaze to other speaker if not currently looking at other speaker
                        wondering_wonder_timer = 0
                        wondering_forced_gaze = False
                        time_within_patch = 0
                # 2. if the other speaker is not looking at us, we start forced gaze if they are looking at us
                else:
                    # if the other speaker turned to look at us
                    if d_val_neural_other[int(round(t/self.dt))] > 0.5 and self.aversion_probability_other[int(round(t/self.dt))] <= 0.5:
                        wondering_forced_gaze = True
                        time_within_patch = 0
                # implement the state machine
                # state 1 implementation, look back at the speaker  
                if wondering_forced_gaze:
                    if output_target[-1] != self.scene_info.other_speaker_id:
                        output_target.append(self.scene_info.other_speaker_id)
                        output_t.append(t)
                    wondering_wonder_timer += self.dt
                # state 2 implementation, wondering around
                else:
                    time_within_patch = t - output_t[-1]
                    # use the random walk algorithm to determine whether to switch patch
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    # make sure the other speaker is not being considered as a look-at-target 
                    look_at_mask[self.conversation_partner_id] = 1
                    not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                    # compute the distance to the patch (first use this variable to store the position of look_at_point)
                    distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                    distance_to_patch = (distance_to_patch - normalized_object_positions)
                    distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                    # compute distance-weighted patch value rho
                    rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                    # compute Q, the expected return of leaving the current patch and move to another patch
                    Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    # compute g_patch, the instetaneous gain by staying at the current patch
                    if rho[current_target] > 0:
                        g_patch = rho[current_target] * np.exp(-self.phi / self.values[index, current_target] * time_within_patch)
                    else:
                        g_patch = 0
                    # compute the probability of migration (logistic function as per the paper, however it is very noisy )
                    p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                    ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                    # if the sampling determine that there is a patch switch
                    if p_stay <= 0.2:
                        rv = np.random.binomial(1, p_stay)
                    else:
                        rv = 1
                    # see where is the nearest next impulse
                    if output_target[-1] == self.scene_info.other_speaker_id or (rv == 0 and time_within_patch >= self.min_gaze_shift_time): # we have to make sure the motion is not too rapid
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(t)
                    # if there isn't a deep learning predicted change
                    current_target = output_target[-1]
            # the case that the neural model does not predict change
            else:
                # if it's not currently a scripted change, then see if the current active object is the other speaker
                # if it is another speaker
                current_target = output_target[-1]
                if self.scene_info.active_object_id == self.scene_info.other_speaker_id:
                    index = int(round(t/self.dt))
                    index = min(index, self.values.shape[0]-1)
                    # see if there is a change in aversion/direct gaze status
                    if d_val_neural[index] > 0.5:
                        if self.aversion_probability[index] <= 0.5:
                            # if it's direct gaze
                            in_aversion = False
                            output_target.append(self.scene_info.other_speaker_id)
                            output_t.append(t)
                        else:
                            # if it's entering aversion
                            in_aversion = True
                            # record the start and end of the aversion
                            aversion_start = t
                            for i in range(index+1, d_val_neural.shape[0]):
                                if d_val_neural[i] > 0.5:
                                    aversion_end = i * self.dt
                            aversion_time = aversion_end - aversion_start
                            # compute rho (value) to determine gaze target
                            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                            look_at_mask[current_target] = 1
                            if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                                pass
                            else:
                                look_at_mask[self.conversation_partner_id] = 1
                            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                            # compute the distance to the patch (first use this variable to store the position of look_at_point)
                            distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                            distance_to_patch = (distance_to_patch - normalized_object_positions)
                            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                            # compute distance-weighted patch value rho
                            rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                    else:
                        # ================================ NEW ================================ 
                        if current_target == self.conversation_partner_id:
                            t += self.dt
                            time_within_patch = t - output_t[-1]
                            continue
                        # ================================ NEW ================================
                        time_within_patch = t - output_t[-1]
                        # use the random walk algorithm to determine whether to switch patch
                        # compute rho (value)
                        look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                        look_at_mask[current_target] = 1
                        if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                            pass
                        else:
                            look_at_mask[self.conversation_partner_id] = 1
                        not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                        # compute the distance to the patch (first use this variable to store the position of look_at_point)
                        distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                        distance_to_patch = (distance_to_patch - normalized_object_positions)
                        distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                        # compute distance-weighted patch value rho
                        rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[current_target] > 0:
                            g_patch = rho[current_target] * np.exp(-self.phi / self.values[index, current_target] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        # if the sampling determine that there is a patch switch
                        if p_stay <= 0.2:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 1
                        # see where is the nearest next impulse
                        if rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time: # we have to make sure the motion is not too rapid
                            # TODO: make the new patch randomly sampled instead of deterministicand
                            # new_patch = np.argmax(rho * not_looked_at_mask)
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            time_within_patch = 0
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                            in_aversion = True
                    # if there isn't a deep learning predicted change
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    # if there isn't an explicit change in aversion status.
                # if the person is just looking into the blank space
                # else:
                    
                #     # if there isn't a deep learning predicted change
                #     current_target = output_target[-1]
                #     time_within_patch = t - output_t[-1]
            t += self.dt
        return output_t, output_target

class Responsive_planner_ThreeParty:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, aversion_probability, aversion_probability_other_l, aversion_probability_other_r, speech_activity, beats, l_id, r_id, self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 2.5
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        self.scripted_gaze_target = saliency_maps[0].gaze_target_over_time
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info:AgentInfo_final_multiparty = scene
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size to padd add the salience map
        self.self_id = self_id
        self.l_id = l_id
        self.r_id = r_id
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
        # pad the saliency maps that are too short
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
        self.object_positions = self.scene_info.get_all_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_frome_positions(self.object_positions) / 180 * np.pi
        # get the conversation partner's id
        self.conversation_partner_id = self.scene_info.active_object_id
        self.aversion_probability = aversion_probability
        self.aversion_probability_other_l = aversion_probability_other_l
        self.aversion_probability_other_r = aversion_probability_other_r
        self.speech_activity = speech_activity # 
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_saccade_time_consecutive
    def get_object_positions(self, current_gaze_target):
        self.scene_info.change_speaker(current_gaze_target)
        positions = self.scene_info.get_all_positions()
        return rotation_angles_frome_positions(positions) / 180 * np.pi
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 2:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = []
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        # compute the look-at-points
        output_target = [self.scene_info.active_object_id]
        output_t = [0]
        normalized_object_positions = self.object_positions
        # simulation time t
        t = 0
        speech_change_counter = 0 # current position in the speech_change_boundaries array         
        gaze_target_counter = 0 # current position in the primary_gaze_target array 
        swap_target = False # whether we are switching between targets
        in_aversion = False # starts out side of gaze aversions
        no_more_gaze_swaps = False
        aversion_start = -1
        aversion_end = -1
        aversion_time = -1
        current_target = output_target[-1]
        length_of_scripted_target = len(self.scripted_gaze_target)
        while t < self.aversion_probability.shape[0] * self.dt + 5:
            # get the next speaker at t + 4
            index = int(np.floor(t/self.dt))
            if index < self.speech_activity.shape[0] - 4:
                next_speaker_in_a_bit = self.speech_activity[index + 4]
            else:
                next_speaker_in_a_bit = self.self_id
            next_speaker_object_id = self.scene_info.get_object_id_based_on_speaker_id(next_speaker_in_a_bit)
            
            # get the next eventual speaker object id (regardless of t+4):
            next_speaker_timeeeee = index
            next_eventual_speaker_object_id = output_target[-1]
            current_speaker_idddddd = -self.self_id
            if index < self.speech_activity.shape[0]:
                current_speaker_idddddd = self.speech_activity[index]
                for i in range(index + 1, self.speech_activity.shape[0]):
                    if self.speech_activity[i] != self.speech_activity[i-1]:
                        next_speaker_timeeeee = i
                        next_eventual_speaker_object_id =  self.scene_info.get_object_id_based_on_speaker_id(self.speech_activity[i])
                        break
            
            # if we are at a scripted target change      
            if length_of_scripted_target > 0 and t > self.scripted_gaze_target[gaze_target_counter][1] - 0.3 and not no_more_gaze_swaps:
                if gaze_target_counter == len(self.scripted_gaze_target) - 1:
                    no_more_gaze_swaps = True
                # see if the active object changes
                if self.scene_info.active_object_id != self.scripted_gaze_target[gaze_target_counter][0]:
                    swap_target = True # if it does, mark the swap_target variable to True
                # change the active object ID
                self.scene_info.active_object_id = self.scripted_gaze_target[gaze_target_counter][0]
                if gaze_target_counter < len(self.scripted_gaze_target) - 1:
                    gaze_target_counter = gaze_target_counter + 1
                if swap_target:
                    output_target.append(self.scene_info.active_object_id)
                    output_t.append(t)
                    swap_target = False
            # the case that the next speaker is not the current gaze target and the next speaker is not the self
            elif next_speaker_object_id != output_target[-1] and next_speaker_in_a_bit != self.self_id:
                # find the object id of that speaker
                current_target = self.scene_info.get_object_id_based_on_speaker_id(next_speaker_in_a_bit)
                output_target.append(current_target)
                output_t.append(t)
            # if we are speaking, we need to look at the next speaker
            elif current_speaker_idddddd == self.self_id and next_eventual_speaker_object_id != output_target[-1]:
                output_target.append(next_eventual_speaker_object_id)
                output_t.append(t)
            # the case that the neural model predicts changes
            else:
                # if it's not currently a scripted change, then see if the current active object is the other speaker
                # if it is another speaker
                current_target = output_target[-1]
                if True or self.scene_info.active_object_id in self.scene_info.other_speaker_id:
                    index = int(round(t/self.dt))
                    index = min(index, self.values.shape[0]-1)
                    # see if there is a change in aversion/direct gaze status
                    if d_val_neural[index] > 0.5:
                        if self.aversion_probability[index] <= 0.5:
                            # if it's direct gaze 
                            in_aversion = False
                            # get the current speaker id:
                            window_of_speaker = self.speech_activity[index:index+25]
                            current_speaker_id = window_of_speaker[0]
                            next_target = -1
                            # we choose the next target to be either the current speaker
                            if current_speaker_id != self.self_id:
                                next_target = current_speaker_id
                            # or the immediate next speaker
                            else:
                                for i_in_win in range(len(window_of_speaker)):
                                    if window_of_speaker[i_in_win] != window_of_speaker[i_in_win-1]:
                                        
                                        next_target = self.scene_info.other_speaker_id[window_of_speaker[i_in_win]]
                                        break
                            if next_target == -1:
                                # if the speaker is speaking himself. Then the speaker will look at the two speaker with an unequal probability
                                speakers = self.scene_info.other_speaker_id
                                # find the previous speaker
                                for jjj in range(0, len(output_target)):
                                    if output_target[-1-jjj] in self.scene_info.other_speaker_id:
                                        next_target = output_target[-1-jjj]  
                            output_target.append(next_target)
                            output_t.append(t)
                        else:
                            # if it's entering aversion
                            in_aversion = True
                            # record the start and end of the aversion
                            aversion_start = t
                            for i in range(index+1, d_val_neural.shape[0]):
                                if d_val_neural[i] > 0.5:
                                    aversion_end = i * self.dt
                            aversion_time = aversion_end - aversion_start
                            # compute rho (value) to determine gaze target
                            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                            look_at_mask[current_target] = 1
                            # if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                            #     pass
                            # else:
                            look_at_mask[self.conversation_partner_id] = 1
                            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                            # compute the distance to the patch (first use this variable to store the position of look_at_point)
                            distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                            distance_to_patch = (distance_to_patch - normalized_object_positions)
                            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                            # compute distance-weighted patch value rho
                            rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                    else:
                        time_within_patch = t - output_t[-1]
                        # use the random walk algorithm to determine whether to switch patch
                        # compute rho (value)
                        look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                        look_at_mask[current_target] = 1
                        # if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                        #     pass
                        # else:
                        look_at_mask[self.conversation_partner_id] = 1
                        not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                        # compute the distance to the patch (first use this variable to store the position of look_at_point)
                        distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                        distance_to_patch = (distance_to_patch - normalized_object_positions)
                        distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                        # compute distance-weighted patch value rho
                        rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[current_target] > 0:
                            g_patch = rho[current_target] * np.exp(-self.phi / self.values[index, current_target] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        # if the sampling determine that there is a patch switch
                        if p_stay <= 0.2:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 1
                        # see where is the nearest next impulse
                        if rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time: # we have to make sure the motion is not too rapid
                            # TODO: make the new patch randomly sampled instead of deterministicand
                            # new_patch = np.argmax(rho * not_looked_at_mask)
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            time_within_patch = 0
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                            in_aversion = True
                    # if there isn't a deep learning predicted change
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    # if there isn't an explicit change in aversion status.
                # if the person is just looking into the blank space
                # else:
                    
                #     # if there isn't a deep learning predicted change
                #     current_target = output_target[-1]
                #     time_within_patch = t - output_t[-1]
            t += self.dt
        return output_t, output_target
        

class Responsive_planner_React_to_gaze_no_Gaze_deploy:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, aversion_probability, aversion_probability_other, speech_activity, beats, self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
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
        # pad the saliency maps that are too short
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
        self.aversion_probability = aversion_probability
        self.aversion_probability_other = aversion_probability_other
        self.speech_activity = speech_activity
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_saccade_time_consecutive
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 2:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = []
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        # compute the look-at-points
        output_target = [self.conversation_partner_id]
        output_t = [0]
        normalized_object_positions = self.object_positions
        in_aversion = False
        for i in range(0, len(speech_change_boundaries)):
            # see if the next state is aversion or direct gaze
            if speech_change_boundaries[i][1] == 0:
                # if it's direct gaze
                in_aversion = False
                output_target.append(self.conversation_partner_id)
                output_t.append(speech_change_boundaries[i][0])
            else:
                # if it's aversion
                if i == len(speech_change_boundaries)-1:
                    aversion_end = speech_change_boundaries[i][0] + 2
                else:
                    aversion_end = speech_change_boundaries[i+1][0]
                aversion_start = speech_change_boundaries[i][0]
                aversion_time = aversion_end - aversion_start
                t = aversion_start
                while t < aversion_end:
                    idx = int(t / self.dt)
                    # make sure index is not out of range
                    idx = min(idx, self.values.shape[0]-1)
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    ##############################################################################################
                    ############################### decide whether to switch patch ###############################
                    ##############################################################################################
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    if self.speech_activity[idx] == 1 and self.aversion_probability_other[idx] == 1:
                        pass
                    else:
                        look_at_mask[self.conversation_partner_id] = 1
                    not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                    # compute the distance to the patch (first use this variable to store the position of look_at_point)
                    distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                    distance_to_patch = (distance_to_patch - normalized_object_positions)
                    vertical_distance = (normalized_object_positions - distance_to_patch)[:, 1]
                    distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                    # compute distance-weighted patch value rho
                     
                    rho = self.values[idx] * np.exp(-self.kappa * distance_to_patch * (1 / aversion_time))
                    # compute Q, the expected return of leaving the current patch and move to another patch
                    Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    # compute g_patch, the instetaneous gain by staying at the current patch
                    if rho[current_target] > 0:
                        g_patch = rho[current_target] * np.exp(-self.phi / self.values[idx, current_target] * time_within_patch)
                    else:
                        g_patch = 0
                    # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                    p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                    ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                    # if the sampling determine that there is a patch switch
                    if p_stay <= 0.2:
                        rv = np.random.binomial(1, p_stay)
                    else:
                        rv = 1
                    # see where is the nearest next impulse
                    if ((rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time 
                         ) or in_aversion == False) and time_within_patch >= self.min_saccade_time_psysiological: # we have to make sure the motion is not too rapid
                        # TODO: make the new patch randomly sampled instead of deterministicand
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(t)
                        in_aversion = True
                    t = t + self.dt
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target

class Responsive_planner_no_Gaze_deploy:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, aversion_probability, beats, self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
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
        # pad the saliency maps that are too short
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
        self.aversion_probability = aversion_probability
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_gaze_shift_time
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 3:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + (self.saliency_maps_arrs[i])
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = []
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        # compute the look-at-points
        output_target = [self.conversation_partner_id]
        output_t = [0]
        normalized_object_positions = self.object_positions
        for i in range(0, len(speech_change_boundaries)):
            # see if the next state is aversion or direct gaze
            if speech_change_boundaries[i][1] == 0:
                # if it's direct gaze
                output_target.append(self.conversation_partner_id)
                output_t.append(speech_change_boundaries[i][0])
            else:
                # if it's aversion
                if i == len(speech_change_boundaries)-1:
                    aversion_end = speech_change_boundaries[i][0] + 2
                else:
                    aversion_end = speech_change_boundaries[i+1][0]
                aversion_start = speech_change_boundaries[i][0]
                aversion_time = aversion_end - aversion_start
                t = aversion_start
                while t < aversion_end:
                    idx = int(t / self.dt)
                    # make sure index is not out of range
                    idx = min(idx, self.values.shape[0]-1)
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    ##############################################################################################
                    ############################### decide whether to switch patch ###############################
                    ##############################################################################################
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    look_at_mask[self.conversation_partner_id] = 1
                    not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                    # compute the distance to the patch (first use this variable to store the position of look_at_point)
                    distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                    distance_to_patch = (distance_to_patch - normalized_object_positions)
                    vertical_distance = (normalized_object_positions - distance_to_patch)[:, 1]
                    distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                    # compute distance-weighted patch value rho
                     
                    rho = self.values[idx] * np.exp(-self.kappa * distance_to_patch * (1 / aversion_time))
                    # compute Q, the expected return of leaving the current patch and move to another patch
                    Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    # compute g_patch, the instetaneous gain by staying at the current patch
                    if rho[current_target] > 0:
                        g_patch = rho[current_target] * np.exp(-self.phi / self.values[idx, current_target] * time_within_patch)
                    else:
                        g_patch = 0
                    # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                    p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                    ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                    # if the sampling determine that there is a patch switch
                    if p_stay <= 0.2:
                        rv = np.random.binomial(1, p_stay)
                    else:
                        rv = 1
                    # see where is the nearest next impulse
                    if ((rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time 
                         ) or current_target==self.conversation_partner_id) and time_within_patch >= self.min_saccade_time_psysiological: # we have to make sure the motion is not too rapid
                        # TODO: make the new patch randomly sampled instead of deterministicand
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(t)
                    t = t + self.dt
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target
class Responsive_planner_no_heuristics:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, self_id=-1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.aversion_comfy_threshold = 3 # need to look back at the parter after 6 seconds
        self.min_eye_contact_threshold = 2 # how short should an average eye contact be 
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.phi = 1 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.05 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
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
        # pad the saliency maps that are too short
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
        d_val = np.abs(dx_dt(self.values, method=2))
        d_val = d_val.sum(axis=1) 
        # compute the look-at-points
        output_target = [self.conversation_partner_id]
        output_t = [0]
        idx = 0
        normalized_object_positions =  self.object_positions
        # try and see when all the change in salience are:
        while idx < self.values.shape[0]-1:
            t = idx* self.dt
            current_target = output_target[-1]
            highest_salience = np.argmax(self.values[idx + 1])
            prev_t = output_t[-1] # previous gaze shift happened at prev_t      
            # if there is a change in the salience list    
            if d_val[idx] >= 0.2:
                # otherwise, if we have been looking away, and need to look at the
                # conversation partner, we look at the partner. 
                if (highest_salience == self.conversation_partner_id and 
                current_target != self.conversation_partner_id):
                    output_target.append(highest_salience)
                    output_t.append(t)
                else:  
                    time_within_patch = t - output_t[-1]
                    ##############################################################################################
                    ############################### decide whether to switch patch ###############################
                    ##############################################################################################
                    # compute rho (value)
                    look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                    look_at_mask[current_target] = 1
                    look_at_mask[self.conversation_partner_id] = 1
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
                    # see where is the nearest next impulse
                    if rv == 0 and time_within_patch >= self.min_saccade_time: # we have to make sure the motion is not too rapid
                        # TODO: make the new patch randomly sampled instead of deterministic
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(self.dt * idx)
            idx = idx + 1
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target
class Responsive_planner_simple:
    # this planner is designed to be responsive to the input heatmap signal
    def __init__(self, saliency_maps, scene, beats, self_id=-1):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.aversion_comfy_threshold = 3 # need to look back at the parter after 6 seconds
        self.min_eye_contact_threshold = 2 # how short should an average eye contact be 
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 1.33
        self.phi = 1 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.4 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        self.beats = beats
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
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if self.beats[i] - t <= 0.02:
                return True
            else:
                return False
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
                current_target != self.conversation_partner_id and self.onbeats(t)):
                    if t - prev_t >= self.min_saccade_time: # we have to make sure the motion is not too rapid
                        output_target.append(highest_salience)
                        output_t.append(t)
                # elif current_target != self.conversation_partner_id and t - prev_t >= self.aversion_comfy_threshold:
                #     output_target.append(self.conversation_partner_id)
                #     output_t.append(t)
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
                    if rv == 0 and time_within_patch >= self.min_saccade_time and self.onbeats(t):
                        # TODO: make the new patch randomly sampled instead of deterministic
                        # new_patch = np.argmax(rho * not_looked_at_mask)
                        new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                        time_within_patch = 0
                        current_target = new_patch
                        output_target.append(current_target)
                        output_t.append(self.dt * idx)
            elif current_target != self.conversation_partner_id and highest_salience == self.conversation_partner_id and t - prev_t >= self.aversion_comfy_threshold:
                if t - prev_t >= self.min_saccade_time: # we have to make sure the motion is not too rapid
                    output_target.append(highest_salience)
                    output_t.append(t)
            idx = idx + 1
        output_t.append(self.dt * self.values.shape[0])
        output_target.append(self.conversation_partner_id)
        return output_t, output_target