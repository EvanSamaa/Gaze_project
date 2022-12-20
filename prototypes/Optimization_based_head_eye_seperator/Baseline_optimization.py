import numpy as np
from matplotlib import pyplot as plt
from Geometry_Util import rotation_angles_frome_positions, directions_from_rotation_angles
from prototypes.Jin2019.EyeHeadDecomposition import Heuristic_decomposition_azimuth, Heuristic_decomposition_elevation, GMM_Decomposition
import cvxpy as cp


def optimize_for_head_gaze_breakdown(gaze_intervals, list_of_gaze_positions, listener_position):

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
    gaze_angles = gaze_angles

    azi_decomp = GMM_Decomposition.fromfile(
        "prototypes/Jin2019/model/head_eye_decomposition_azimuth_60_clusters_fixation/")
    ele_decomp = GMM_Decomposition.fromfile(
        "prototypes/Jin2019/model/head_eye_decomposition_elevation_60_clusters_fixation/")

    # motion prior
    motion_priors = []
    # get the prior angles one by one
    prior_head_angles = []
    for i in range(0, gaze_angles.shape[0]):
        prev_azi = gaze_angles[max(0, i-1), 0]
        azi_angle = gaze_angles[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, prev_azi)
        prev_ele = gaze_angles[max(0, i-1), 1]
        ele_angle = gaze_angles[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, prev_ele)
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_head_angles.append((new_head_angles))

    prior_head_angles = np.concatenate(prior_head_angles, axis=0)

    # center bias is the tendency of looking at the center
    center_bias_head_angles = np.zeros((prior_head_angles.shape))

    # listener bias is the tendency of looking at the listner
    listener_angle = rotation_angles_frome_positions(listener_position)

    # solve for the neck based on the simple optimization
    solved_angles = []
    for i in range(0, prior_head_angles.shape[0]):
        gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
        neck_angle_ele = cp.Variable(1)
        neck_angle_azi = cp.Variable(1)
        # optimize for neck angle azi
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
        #                         + (neck_angle_azi - listener_angle[0])**2)
        objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        # optimize for neck angle elevation
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2
        #                         + (neck_angle_ele - listener_angle[1])**2)
        objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_ele - prior_head_angles[i, 1]) ** 2)
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        solved_angles.append(np.array([[neck_angle_azi.value[0], neck_angle_ele.value[0]]]))
    solved_angles = np.concatenate(solved_angles, axis = 0)
    head_pos = directions_from_rotation_angles(solved_angles, gaze_positions_norm)
    return head_pos









