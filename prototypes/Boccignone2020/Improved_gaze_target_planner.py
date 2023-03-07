from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import AversionSignalDrivenSaliency
from Geometry_Util import rotation_angles_frome_positions
import numpy as np
from matplotlib import pyplot as plt
class Scavenger_planner_with_nest:
    def __init__(self, saliency_maps, scene_info, self_id=-1):
        # hyper-parameters
        self.smoothing_constant = 0.2
        self.kappa = 1.3333333 # this is the distance factor (i.e. cost of migration), this is from the paper
        self.kappa = 2.2 # this is the distance factor (i.e. cost of migration)
        self.momentum_weight = 2
        self.phi = .5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.2 # this specified how closely two nearby saccade can be with one another.

        self.nest_consumption_rate = 0.5 # the amount of time to consume food at the nest
        # TODO: this tao should also be time varying (Using this for the basis to implement reactive gaze (to the
        # listener's gaze))
        self.predation_risk_tao = 0.5 # the constant for exponential distribution for predation
        # get the dt
        self.dt = saliency_maps[0]._dt
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info = scene_info
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size
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
        # get the position of objects
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_frome_positions(self.object_positions) / 180 * np.pi
        # get the conversation partner's id
        if self.self_id == -1:
            self.conversation_partner_id = saliency_maps[0].scene_info.get_conversation_partner_id()[0]
            # ========================================= state variables ==========================================
            self.current_look_at = self.conversation_partner_id
            # index for the nest, of which the gaze will gravitate towards
            self.nest_index = self.conversation_partner_id
        else:
            self.conversation_partner_id = saliency_maps[0].scene_info.get_conversation_partner_id(self.self_id, 0)
            # ========================================= state variables ==========================================
            self.current_look_at = self.conversation_partner_id
            # index for the nest, of which the gaze will gravitate towards
            self.nest_index = self.conversation_partner_id
        # momentum stores the current gaze_direction from the conversation-partner
        self.momentum = np.array([0, 0])
    def compute(self, initial_target):
        # variable to store the output values
        output_t = [0]
        output_target = [self.nest_index]
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
        if self.self_id >= 0:
            values[:, self.self_id] = 0
        # initialize the first look at point with the user speficied initial target
        self.current_look_at = self.nest_index
        time_within_patch = 0
        time_away_from_nest = 0
        # add the first target to the output list
        output_target.append(self.current_look_at)
        output_t.append(0)
        for i in range(0, self.saliency_maps_arrs[0].shape[0]):
            # update new nest
            if self.self_id >= 0:
                self.nest_index = self.scene_info.get_conversation_partner_id(self.self_id, i*self.dt)
            ##############################################################################################
            ############################### decide whether to switch patch ###############################
            ##############################################################################################
            # compute rho (value)
            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
            look_at_mask[self.current_look_at] = 1
            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
            # compute the distance to the patch (first use this variable to store the position of look_at_point)
            distance_to_patch = np.tile(self.object_positions[self.current_look_at:self.current_look_at+1], [self.object_positions.shape[0], 1])
            distance_to_patch = (distance_to_patch - self.object_positions)
            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))

            # compute the distance to the nest
            distance_to_nest = np.tile(self.object_positions[self.nest_index:self.nest_index + 1], [self.object_positions.shape[0], 1])
            distance_to_nest = (distance_to_nest - self.object_positions)
            distance_to_nest = np.sqrt(np.square(distance_to_nest).sum(axis=1))
            # updates 5 times a second
            if (float(i) * self.dt / 0.1).is_integer() and i > 0:
                # if we are currently in the nest
                if self.current_look_at == self.nest_index :
                    # compute the average value of going out
                    M = self.nest_consumption_rate
                    rho = values[i] * np.exp(-self.kappa * distance_to_patch)
                    rho[self.current_look_at] = rho[self.current_look_at] * np.exp(-self.nest_consumption_rate * time_within_patch)
                    rho_mean = 1 / (rho.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    rho_max = np.max(rho * not_looked_at_mask)
                    p_leave = 1 / (1 + np.exp(self.beta * (rho[self.current_look_at] - rho_mean)))
                    rv = np.random.binomial(1, p_leave)
                    if rv == 1:
                        prob = values[i] * np.exp(-self.kappa * distance_to_patch) * not_looked_at_mask
                        heat = self.beta/2
                        probability = np.exp(heat * prob)/np.sum(np.exp(heat * prob))
                        probability[self.self_id] = 0
                        probability = probability / probability.sum()
                        # find the item with maximum probability
                        deterministic_new_patch = np.argmax(prob)
                        # sample the items for a more randomized new look-at-point
                        sampled_new_patch = np.random.choice(np.arange(0, prob.shape[0]), 1, p=probability)[0]
                        # use the sampled patch id for better looking result in a static scene
                        new_patch = sampled_new_patch
                        time_within_patch = 0
                        time_away_from_nest = 0
                        self.current_look_at = new_patch
                        output_target.append(self.current_look_at)
                        output_t.append(self.dt * i)
                        self.momentum = self.object_positions[self.current_look_at] - self.object_positions[self.nest_index]
                else:

                    # compute distance-weighted patch value rho
                    momemtum_distance = np.expand_dims(self.momentum, axis=0)
                    momemtum_distance = momemtum_distance * (self.object_positions - np.expand_dims(self.object_positions[self.nest_index], axis=0))
                    momemtum_distance = 1 / (1 + np.exp(- 10 * momemtum_distance.sum(axis=1)))
                    rho = values[i] * np.exp(-self.kappa * distance_to_patch)
                    risk = np.exp(-self.kappa * distance_to_nest) * self.predation_risk_tao * np.exp(self.predation_risk_tao * time_away_from_nest)
                    # if it is still worth it to not return to nest
                    if (rho - risk).max() > 0:
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask * np.exp(-momemtum_distance))
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[self.current_look_at] > 0:
                            g_patch = rho[self.current_look_at]
                            g_patch = rho[self.current_look_at] * np.exp(-self.phi / values[i, self.current_look_at] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        # if the sampling determine that there is a patch switch (the issue is that given the sampling frequency,
                        # it is actually highly likely for the gaze target to switch
                        if p_stay <= 1.0:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 0
                        if rv == 0 and time_within_patch >= self.min_saccade_time:
                            # get probability of the gaze aversion
                            prob = values[i] * np.exp(-self.kappa * distance_to_patch) * not_looked_at_mask * np.exp(-self.momentum_weight*momemtum_distance)
                            heat = self.beta / 2
                            probability = np.exp(heat * prob) / np.sum(np.exp(heat * prob))
                            probability[self.self_id] = 0
                            probability = probability / probability.sum()
                            # find the item with maximum probability
                            deterministic_new_patch = np.argmax(prob)
                            # sample the items for a more randomized new look-at-point
                            sampled_new_patch = np.random.choice(np.arange(0, prob.shape[0]), 1, p=probability)[0]
                            # use the sampled patch id for better looking result in a static scene
                            new_patch = sampled_new_patch

                            time_within_patch = 0
                            self.current_look_at = new_patch
                            output_target.append(self.current_look_at)
                            output_t.append(self.dt * i)
                            self.momentum = self.object_positions[self.current_look_at] - self.object_positions[self.nest_index]
                    else:
                        time_within_patch = 0
                        time_away_from_nest = 0
                        self.current_look_at = self.nest_index
                        output_target.append(self.current_look_at)
                        output_t.append(self.dt * i)
                        self.momentum = self.momentum * 0

            # accumulate time away and within patch
            time_within_patch += self.dt
            time_away_from_nest += self.dt
        return output_t, output_target
    def model_fitting(self, average_aversion_duration, average_gaze_duration):
        return
class Scavenger_planner_simple:
    def __init__(self, saliency_maps, scene_info, self_id=-1):
        # hyper-parameters
        self.smoothing_constant = 0.2
        self.kappa = 1.333333 # this is the distance factor (i.e. cost of migration), this is from the paper
        self.kappa = 2.2 # this is the distance factor (i.e. cost of migration)
        self.momentum_weight = 3
        self.phi = .5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time = 0.4 # this specified how closely two nearby saccade can be with one another.

        self.nest_consumption_rate = 0.05 # the amount of time to consume food at the nest
        # TODO: this tao should also be time varying (Using this for the basis to implement reactive gaze (to the
        # listener's gaze))
        self.predation_risk_tao = 0.5 # the constant for exponential distribution for predation
        # get the dt
        self.dt = saliency_maps[0]._dt
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info = scene_info
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size
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
        # get the position of objects
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_frome_positions(self.object_positions) / 180 * np.pi
        # get the conversation partner's id
        if self.self_id == -1:
            self.conversation_partner_id = agentScene1.object_pos.shape[0]
            # ========================================= state variables ==========================================
            self.current_look_at = self.conversation_partner_id
            # index for the nest, of which the gaze will gravitate towards
            self.nest_index = self.conversation_partner_id
        else:
            self.conversation_partner_id = agentScene1.object_pos.shape[0]
            # ========================================= state variables ==========================================
            self.current_look_at = self.conversation_partner_id
            # index for the nest, of which the gaze will gravitate towards
            self.nest_index = self.conversation_partner_id
        # momentum stores the current gaze_direction from the conversation-partner
        self.momentum = np.array([0, 0])
    def compute(self, initial_target):
        # variable to store the output values
        output_t = [0]
        output_target = [self.nest_index]
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
        if self.self_id >= 0:
            values[:, self.self_id] = 0
        # initialize the first look at point with the user speficied initial target
        self.current_look_at = self.nest_index
        time_within_patch = 0
        time_away_from_nest = 0
        # add the first target to the output list
        output_target.append(self.current_look_at)
        output_t.append(0)
        for i in range(0, self.saliency_maps_arrs[0].shape[0]):
            # update new nest
            if self.self_id >= 0:
                self.nest_index = agentScene1.object_pos.shape[0]
            ##############################################################################################
            ############################### decide whether to switch patch ###############################
            ##############################################################################################
            # compute rho (value)
            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
            look_at_mask[self.current_look_at] = 1
            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
            # compute the distance to the patch (first use this variable to store the position of look_at_point)
            distance_to_patch = np.tile(self.object_positions[self.current_look_at:self.current_look_at+1], [self.object_positions.shape[0], 1])
            distance_to_patch = (distance_to_patch - self.object_positions)
            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))

            # compute the distance to the nest
            distance_to_nest = np.tile(self.object_positions[self.nest_index:self.nest_index + 1], [self.object_positions.shape[0], 1])
            distance_to_nest = (distance_to_nest - self.object_positions)
            distance_to_nest = np.sqrt(np.square(distance_to_nest).sum(axis=1))
            # updates 5 times a second
            if (float(i) * self.dt / 0.1).is_integer() and i > 0:
                # if we are currently in the nest
                if self.current_look_at == self.nest_index :
                    # compute the average value of going out
                    M = self.nest_consumption_rate
                    rho = values[i] * np.exp(-self.kappa * distance_to_patch)
                    rho[self.current_look_at] = rho[self.current_look_at] * np.exp(-self.nest_consumption_rate * time_within_patch)
                    rho_mean = 1 / (rho.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                    rho_max = np.max(rho * not_looked_at_mask)
                    p_leave = 1 / (1 + np.exp(self.beta * (rho[self.current_look_at] - rho_mean)))
                    rv = np.random.binomial(1, p_leave)
                    if rv == 1:
                        prob = values[i] * np.exp(-self.kappa * distance_to_patch) * not_looked_at_mask
                        heat = self.beta/2
                        probability = np.exp(heat * prob)/np.sum(np.exp(heat * prob))
                        probability[self.self_id] = 0
                        probability = probability / probability.sum()
                        # find the item with maximum probability
                        deterministic_new_patch = np.argmax(prob)
                        # sample the items for a more randomized new look-at-point
                        sampled_new_patch = np.random.choice(np.arange(0, prob.shape[0]), 1, p=probability)[0]
                        # use the sampled patch id for better looking result in a static scene
                        new_patch = sampled_new_patch
                        time_within_patch = 0
                        time_away_from_nest = 0
                        self.current_look_at = new_patch
                        output_target.append(self.current_look_at)
                        output_t.append(self.dt * i)
                        self.momentum = self.object_positions[self.current_look_at] - self.object_positions[self.nest_index]
                else:

                    # compute distance-weighted patch value rho
                    momemtum_distance = np.expand_dims(self.momentum, axis=0)
                    momemtum_distance = momemtum_distance * (self.object_positions - np.expand_dims(self.object_positions[self.nest_index], axis=0))
                    momemtum_distance = 1 / (1 + np.exp(- 10 * momemtum_distance.sum(axis=1)))
                    rho = values[i] * np.exp(-self.kappa * distance_to_patch)
                    risk = np.exp(-self.kappa * distance_to_nest) * self.predation_risk_tao * np.exp(self.predation_risk_tao * time_away_from_nest)
                    # if it is still worth it to not return to nest
                    if (rho - risk).max() > 0:
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask * np.exp(-momemtum_distance))
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[self.current_look_at] > 0:
                            g_patch = rho[self.current_look_at]
                            g_patch = rho[self.current_look_at] * np.exp(-self.phi / values[i, self.current_look_at] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        # if the sampling determine that there is a patch switch (the issue is that given the sampling frequency,
                        # it is actually highly likely for the gaze target to switch
                        if p_stay <= 1.0:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 0
                        if rv == 0 and time_within_patch >= self.min_saccade_time:
                            # get probability of the gaze aversion
                            prob = values[i] * np.exp(-self.kappa * distance_to_patch) * not_looked_at_mask * np.exp(-self.momentum_weight*momemtum_distance)
                            heat = self.beta / 2
                            probability = np.exp(heat * prob) / np.sum(np.exp(heat * prob))
                            probability[self.self_id] = 0
                            probability = probability / probability.sum()
                            # find the item with maximum probability
                            deterministic_new_patch = np.argmax(prob)
                            # sample the items for a more randomized new look-at-point
                            sampled_new_patch = np.random.choice(np.arange(0, prob.shape[0]), 1, p=probability)[0]
                            # use the sampled patch id for better looking result in a static scene
                            new_patch = sampled_new_patch

                            time_within_patch = 0
                            self.current_look_at = new_patch
                            output_target.append(self.current_look_at)
                            output_t.append(self.dt * i)
                            self.momentum = self.object_positions[self.current_look_at] - self.object_positions[self.nest_index]
                    else:
                        time_within_patch = 0
                        time_away_from_nest = 0
                        self.current_look_at = self.nest_index
                        output_target.append(self.current_look_at)
                        output_t.append(self.dt * i)
                        self.momentum = self.momentum * 0

            # accumulate time away and within patch
            time_within_patch += self.dt
            time_away_from_nest += self.dt
        return output_t, output_target
    def model_fitting(self, average_aversion_duration, average_gaze_duration):
        return
