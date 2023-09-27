import numpy as np
from matplotlib import pyplot as plt
from Geometry_Util import rotation_angles_frome_positions, directions_from_rotation_angles
from prototypes.Jin2019.EyeHeadDecomposition import Heuristic_decomposition_azimuth, Heuristic_decomposition_elevation, GMM_Decomposition
import cvxpy as cp

def get_gaze_target(t, gaze_target_over_time):
    if t > gaze_target_over_time[-1][0]:
        return gaze_target_over_time[-1][1]
    else:
        for i in range(0, len(gaze_target_over_time)-1):
            if t >= gaze_target_over_time[i][0] and t <= gaze_target_over_time[i+1][0]:
                return gaze_target_over_time[i][1]
    return gaze_target_over_time[0][1]
def optimize_for_head_gaze_breakdown_three_party_scene(gaze_intervals, list_of_gaze_positions, internal_model, gaze_target_over_time):
    # listener bias is the tendency of looking at the listner
    listener_position = []
    for i in range(0, len(gaze_intervals)):
        t = gaze_intervals[i][0] + 0.4
        internal_model.new_listener(get_gaze_target(t, gaze_target_over_time))
        # print(internal_model.scene.active_object_id)
        listener_position.append(np.expand_dims(internal_model.estimate_listener_pose(), axis=0))
    listener_position = np.concatenate(listener_position, axis=0)
    
    listener_angle_expand = rotation_angles_frome_positions(listener_position)
    # listener_angle_expand = np.expand_dims(listener_angle, axis=0)
    # gaze_position is probably a list of array
    gaze_positions = []
    for i in range(len(list_of_gaze_positions)):
        gaze_positions.append(np.expand_dims(list_of_gaze_positions[i], axis=0))
    # concatenating the arrays into one 2D array with the shape [N, 3]
    gaze_positions = np.concatenate(gaze_positions, axis=0)

    # get the norm of each positions
    gaze_positions_norm = np.sqrt(np.square(gaze_positions).sum(axis=1))
    # get the angle for each positions (since it's slightly easier to operate)
    gaze_angles = rotation_angles_frome_positions(gaze_positions)
    # get the heuristic head-eye decomposition
    azi_decomp = Heuristic_decomposition_azimuth()
    ele_decomp = Heuristic_decomposition_elevation()
    # motion prior
    motion_priors = [] 
    # get the prior angles one by one
    prior_head_angles = []
    
    for i in range(0, gaze_angles.shape[0]):
        prev_azi = gaze_angles[max(0, i-1), 0]
        azi_angle = gaze_angles[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = gaze_angles[max(0, i-1), 1]
        ele_angle = gaze_angles[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.3)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head * 0.3
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_head_angles.append((new_head_angles))
    prior_head_angles = np.concatenate(prior_head_angles, axis=0)
    # get the prior angle of the long term look-at-target
    same_target = []
    prior_long_term_target_angles = []
    for i in range(0, gaze_angles.shape[0]):
        # if we are looking at the listener then we mark it as True
        if np.linalg.norm(listener_angle_expand[i] - gaze_angles[i]) <= 0.00001:
            same_target.append(True)
        else:
            same_target.append(False)
        prev_azi = listener_angle_expand[max(0, i-1), 0]
        azi_angle = listener_angle_expand[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = listener_angle_expand[max(0, i-1), 1]
        ele_angle = listener_angle_expand[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.3)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_long_term_target_angles.append((new_head_angles))
    prior_long_term_target_angles = np.concatenate(prior_long_term_target_angles, axis=0)
    # center bias is the tendency of looking at the center
    center_bias_head_angles = np.zeros((prior_head_angles.shape))
    # solve for the neck based on the simple optimization
    solved_angles = []
    camera_position = internal_model.scene.get_camera_pos()
    camera_angle = rotation_angles_frome_positions(camera_position)
    prev_angle = [0, 0]
    for i in range(0, prior_head_angles.shape[0]):
        gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
        neck_angle_ele = cp.Variable(1)
        neck_angle_azi = cp.Variable(1)
        # optimize for neck angle azi
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
        #                         + (neck_angle_azi - listener_angle[0])**2)
        gaze_target = gaze_angles[i]
        prior_angle = prior_head_angles[i]
        camera_angle = camera_angle
        prev_angle = prev_angle
        
        # prior_head_angles[i] = prior_head_angles[i] - prev_angle 
        objective = cp.Minimize(0 +
                               1 * (prior_angle[0] - neck_angle_azi) ** 2 +
                                min(gaze_time, 2) * (gaze_target[0] - neck_angle_azi) ** 2 + max(2 - gaze_time, 0) * (neck_angle_azi - prior_long_term_target_angles[i][0]) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # optimize for neck angle elevation
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2
        #                         + (neck_angle_ele - listener_angle[1])**2)
        objective = cp.Minimize(0 +
                                1 * (prior_angle[1] - neck_angle_ele) ** 2 +
                                min(gaze_time, 2) * (gaze_target[1] - neck_angle_ele) ** 2 + 0.1 * (neck_angle_ele - prior_long_term_target_angles[i][1]) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # if same_target[i]:
        #     solved_angles.append(np.array([[prior_long_term_target_angles[i][0], 
        #                                     prior_long_term_target_angles[i][1]]]))
        # else:
        #     solved_angles.append(np.array([[neck_angle_azi.value[0] + prior_long_term_target_angles[i][0], 
        #                                     neck_angle_ele.value[0] + prior_long_term_target_angles[i][1]]]))
        solved_angles.append(np.array([[neck_angle_azi.value[0],  neck_angle_ele.value[0]]]))
        prev_angle = [neck_angle_azi.value[0],  neck_angle_ele.value[0]]
        # prev_angle = np.array([neck_angle_azi.value[0] + prior_long_term_target_angles[i][0], 
        #                        neck_angle_ele.value[0] + prior_long_term_target_angles[i][1]])
        # solved_angles.append(np.array([[listener_angle[0], listener_angle[0]]]))
    solved_angles = np.concatenate(solved_angles, axis = 0)
    head_pos = directions_from_rotation_angles(solved_angles, np.expand_dims(gaze_positions_norm, axis=1))
    return head_pos

def optimize_for_head_gaze_breakdown_dynamic_scene(gaze_intervals, list_of_gaze_positions, internal_model, gaze_target_over_time):
    # listener bias is the tendency of looking at the listner
    listener_position = []
    actual_listener_position = internal_model.scene.get_object_positions(internal_model.scene.other_speaker_id)
    for i in range(0, len(gaze_intervals)):
        t = gaze_intervals[i][0] + 0.4
        internal_model.new_listener(get_gaze_target(t, gaze_target_over_time))
        # print(internal_model.scene.active_object_id)
        listener_position.append(np.expand_dims(internal_model.estimate_listener_pose(), axis=0))
    listener_position = np.concatenate(listener_position, axis=0)
    
    listener_angle_expand = rotation_angles_frome_positions(listener_position)
    # listener_angle_expand = np.expand_dims(listener_angle, axis=0)
    # gaze_position is probably a list of array
    gaze_positions = []
    for i in range(len(list_of_gaze_positions)):
        gaze_positions.append(np.expand_dims(list_of_gaze_positions[i], axis=0))
    # concatenating the arrays into one 2D array with the shape [N, 3]
    gaze_positions = np.concatenate(gaze_positions, axis=0)

    # get the norm of each positions
    gaze_positions_norm = np.sqrt(np.square(gaze_positions).sum(axis=1))
    # get the angle for each positions (since it's slightly easier to operate)
    gaze_angles = rotation_angles_frome_positions(gaze_positions)
    # get the heuristic head-eye decomposition
    azi_decomp = Heuristic_decomposition_azimuth()
    ele_decomp = Heuristic_decomposition_elevation()
    # motion prior
    motion_priors = [] 
    # get the prior angles one by one
    prior_head_angles = []
    
    for i in range(0, gaze_angles.shape[0]):
        prev_azi = gaze_angles[max(0, i-1), 0]
        azi_angle = gaze_angles[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = gaze_angles[max(0, i-1), 1]
        ele_angle = gaze_angles[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.8)
        # make the character look down less if they are looking down
        # if ele_head < 0:
        #     ele_head = ele_head * 0.3
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_head_angles.append((new_head_angles))
    prior_head_angles = np.concatenate(prior_head_angles, axis=0)
    # get the prior angle of the long term look-at-target
    same_target = []
    prior_long_term_target_angles = []
    for i in range(0, gaze_angles.shape[0]):
        # if we are looking at the listener then we mark it as True
        if np.linalg.norm(listener_angle_expand[i] - gaze_angles[i]) <= 0.00001:
            same_target.append(True)
        else:
            same_target.append(False)
        prev_azi = listener_angle_expand[max(0, i-1), 0]
        azi_angle = listener_angle_expand[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = listener_angle_expand[max(0, i-1), 1]
        ele_angle = listener_angle_expand[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.3)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_long_term_target_angles.append((new_head_angles))
    prior_long_term_target_angles = np.concatenate(prior_long_term_target_angles, axis=0)
    # center bias is the tendency of looking at the center
    center_bias_head_angles = np.zeros((prior_head_angles.shape))
    # solve for the neck based on the simple optimization
    solved_angles = []
    camera_position = internal_model.scene.get_camera_pos()
    camera_angle = rotation_angles_frome_positions(camera_position)
    prev_angle = listener_angle_expand[0]
    for i in range(0, prior_head_angles.shape[0]):
        gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
        neck_angle_ele = cp.Variable(1)
        neck_angle_azi = cp.Variable(1)
        # optimize for neck angle azi
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
        #                         + (neck_angle_azi - listener_angle[0])**2)
        gaze_target = gaze_angles[i] - prior_long_term_target_angles[i]
        prior_angle = prior_head_angles[i] - prior_long_term_target_angles[i]
        camera_angle = camera_angle
        prev_angle = prev_angle
        
        # prior_head_angles[i] = prior_head_angles[i] - prev_angle 
        objective = cp.Minimize(0 +
                                min(gaze_time, 3) * (prior_angle[0] - neck_angle_azi) ** 2 +
                                max(3 - gaze_time, 0) * (neck_angle_azi) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # optimize for neck angle elevation
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2
        #                         + (neck_angle_ele - listener_angle[1])**2)
        objective = cp.Minimize(0 +
                                min(gaze_time, 3) * (prior_angle[1] - neck_angle_ele) ** 2 +
                                max(3 - gaze_time, 0) * (neck_angle_ele) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        if same_target[i]:
            solved_angles.append(np.array([[prior_long_term_target_angles[i][0], 
                                            prior_long_term_target_angles[i][1]]]))
        else:
            solved_angles.append(np.array([[neck_angle_azi.value[0] + prior_long_term_target_angles[i][0], 
                                            neck_angle_ele.value[0] + prior_long_term_target_angles[i][1]]]))
        

        # prev_angle = np.array([neck_angle_azi.value[0] + prior_long_term_target_angles[i][0], 
        #                        neck_angle_ele.value[0] + prior_long_term_target_angles[i][1]])
        # solved_angles.append(np.array([[listener_angle[0], listener_angle[0]]]))
    solved_angles = np.concatenate(solved_angles, axis = 0)
    head_pos = directions_from_rotation_angles(solved_angles, np.expand_dims(gaze_positions_norm, axis=1))
    return head_pos

def optimize_for_head_gaze_breakdown(gaze_intervals, list_of_gaze_positions, listener_position):
    # listener bias is the tendency of looking at the listner
    listener_angle = rotation_angles_frome_positions(listener_position)
    listener_angle_expand = np.expand_dims(listener_angle, axis=0)
    # gaze_position is probably a list of array
    gaze_positions = []
    for i in range(len(list_of_gaze_positions)):
        gaze_positions.append(np.expand_dims(list_of_gaze_positions[i], axis=0))
    # concatenating the arrays into one 2D array with the shape [N, 3]
    gaze_positions = np.concatenate(gaze_positions, axis=0)

    # get the norm of each positions
    gaze_positions_norm = np.sqrt(np.square(gaze_positions).sum(axis=1))
    # get the angle for each positions (since it's slightly easier to operate)
    gaze_angles = rotation_angles_frome_positions(gaze_positions)
    # get the angle to be in degrees
    gaze_angles = gaze_angles - listener_angle_expand
    # azi_decomp = GMM_Decomposition.fromfile(
    #     "prototypes/Jin2019/model/head_eye_decomposition_azimuth_60_clusters_fixation/")
    # ele_decomp = GMM_Decomposition.fromfile(
    #     "prototypes/Jin2019/model/head_eye_decomposition_elevation_60_clusters_fixation/")
    azi_decomp = Heuristic_decomposition_azimuth()
    ele_decomp = Heuristic_decomposition_elevation()
    # motion prior
    motion_priors = [] 
    # get the prior angles one by one
    prior_head_angles = []
    for i in range(0, gaze_angles.shape[0]):
        prev_azi = gaze_angles[max(0, i-1), 0]
        azi_angle = gaze_angles[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.3)
        prev_ele = gaze_angles[max(0, i-1), 1]
        ele_angle = gaze_angles[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.5)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head * 0.3
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_head_angles.append((new_head_angles))

    prior_head_angles = np.concatenate(prior_head_angles, axis=0)
    # center bias is the tendency of looking at the center
    center_bias_head_angles = np.zeros((prior_head_angles.shape))
    prior_head_angles = gaze_angles
    # solve for the neck based on the simple optimization
    solved_angles = []
    for i in range(0, prior_head_angles.shape[0]):
        gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
        neck_angle_ele = cp.Variable(1)
        neck_angle_azi = cp.Variable(1)
        # optimize for neck angle azi
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
        #                         + (neck_angle_azi - listener_angle[0])**2)
        objective = cp.Minimize(0 +
                                min(gaze_time, 1.5) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2 +
                                max(1.5 - gaze_time, 0.00001) * (neck_angle_azi) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # optimize for neck angle elevation
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2
        #                         + (neck_angle_ele - listener_angle[1])**2)
        objective = cp.Minimize(0 +
                                min(gaze_time, 1.5) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2 +
                                max(1.5 - gaze_time, 0) * (neck_angle_ele) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # solved_angles.append(np.array([[prior_head_angles[i, 0]+listener_angle[0], prior_head_angles[i, 1]+listener_angle[0]]]))
        solved_angles.append(np.array([[neck_angle_azi.value[0]+listener_angle[0], neck_angle_ele.value[0]+listener_angle[1]]]))
        # solved_angles.append(np.array([[listener_angle[0], listener_angle[0]]]))

    solved_angles = np.concatenate(solved_angles, axis = 0)
    head_pos = directions_from_rotation_angles(solved_angles, gaze_positions_norm)
    return head_pos









