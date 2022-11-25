import numpy as np
try:
    import prototypes.Jin2019.EyeHeadDecomposition as decomp
except:
    import EyeHeadDecomposition as decomp
def handle_saccade(start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05, avg_saccade_interval=0.5):
    output_list = []
    rig_factor = 1 # I believe it should be rig_factor[130] and rig_factor[131],
    max_saccade_x = 2 * saccade_factor
    max_saccade_y = 2 * saccade_factor
    prev_saccade_frame_counter = prev_saccade_frame
    if prev_saccade_frame <= start_frame:
        saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
        saccade_duration = 1.0/24.0
        output_list.append([start_frame, 0, 0])
        prev_saccade_frame_counter = start_frame + saccade_interval
        output_list.append([prev_saccade_frame_counter, 0, 0])
        prev_saccade_frame_counter += saccade_duration
    while prev_saccade_frame_counter < end_frame:
        # compute offset
        offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
        offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
        saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval/10.0
        saccade_duration = 1.0/24.0
        output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
        prev_saccade_frame_counter += saccade_interval
        output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
        prev_saccade_frame_counter += saccade_duration
    return output_list, prev_saccade_frame_counter
# slower head velocity makes it necessary to add in co-articulation between consecutive gaze shifts

# this version will allow the following parameters to be parameterized
# head velocity (o) -> add a factor for the head
# head amplitude/relative to eye (o) -> need to modify heuristic decomposition functions
# head delay/relative to eye (o) -> need to modify the heuristic decomposition functions
# overall gaze delay (o) -> this is new
# break the intervals into handler functions (o) -> this is nice
# For now, the parameters will be in the format:
# [reaction speed, head contribution, head reactivity, head velocity]
# 1 indicates more, fasters, more reactive and 0 indicates less

# slower head velocity makes it necessary to add in co-articulation between consecutive gaze shifts
def handle_gaze_direction(old_gaze, previous_head_frame, new_gaze, new_time,
                          parameters, az_decomp, ele_decomp, eye_v_decomp):
    head_keyframes = []
    gaze_key_frames = []

    # in case I give the head an unwanted jitter
    if np.linalg.norm(old_gaze - new_gaze) == 0:
        return None, None, None

        # stupidly fast reaction time is 0.1 second, on average it is betweem 0.2-0.3s
    reaction_time = 0.3 - 0.2 * parameters[0]
    reaction_time += np.random.normal(0, 0.005)  # as per literature

    ##############################################################
    ######################### EYESSSS ############################
    ##############################################################
    # compute timing (velocity) for the eye based on heuristics.
    # TODO: maybe add paramterization later
    delta_gaze = np.linalg.norm(old_gaze - new_gaze)
    gaze_speed = eye_v_decomp.decompose(delta_gaze)
    eye_travel_time = delta_gaze / gaze_speed

    # compute actual timings of the keyframes! Woohoo exciting!
    eye_start = float(new_time + reaction_time)
    eye_end = float(eye_start + eye_travel_time)

    # preparing output structure for gaze
    gaze_key_frames.append([eye_start, float(old_gaze[0]), float(old_gaze[1])])
    gaze_key_frames.append([eye_end, float(new_gaze[0]), float(new_gaze[1])])

    ##############################################################
    ######################### BANDAID ############################
    ##############################################################

    # dealing with bad cases of head movements
    if previous_head_frame[-1][0] < new_time:
        old_head_position = np.array([previous_head_frame[-1][1], previous_head_frame[-1][2]])
    else:
        # search through the points to keep only the valid points
        i = len(previous_head_frame) - 1
        while previous_head_frame[i][0] >= new_time:
            i = i - 1
            if i < -1:
                new_previous_head_frame = previous_head_frame[0:1]
                return gaze_key_frames, [], previous_head_frame
        new_previous_head_frame = previous_head_frame[:i + 1]
        t0 = previous_head_frame[i][0]
        t1 = previous_head_frame[i + 1][0]
        x0 = previous_head_frame[i][1]
        y0 = previous_head_frame[i][2]
        x1 = previous_head_frame[i + 1][1]
        y1 = previous_head_frame[i + 1][2]
        old_head_position = np.array([previous_head_frame[-1][1], previous_head_frame[-1][2]])
        x_new = (new_time - t0) / (t1 - t0) * (x1 - x0) + x0
        y_new = (new_time - t0) / (t1 - t0) * (y1 - y0) + y0

        new_previous_head_frame.append([float(new_time), float(x_new),
                                        float(y_new)])
        old_head_position = np.array([float(x_new), float(y_new)])
        previous_head_frame = new_previous_head_frame

    ##############################################################
    ########################### NECK #############################
    ##############################################################

    # compute head amplitude
    new_eye_angle_az, new_head_angle_az = az_decomp.decompose(new_gaze[0], parameters[1])
    new_eye_angle_ele, new_head_angle_ele = ele_decomp.decompose(new_gaze[1])

    # compute head timing (velocity)
    new_head_angle = np.array([new_head_angle_az, new_head_angle_ele])
    delta_head = np.linalg.norm(new_head_angle - old_head_position)
    head_speed = min((2 + (parameters[3]) * 4) * delta_head + 20, 300)
    head_travel_time = delta_head / head_speed

    # compute head movement delay (could be very slow could be fast)
    head_delay = 0 + 0.05 * parameters[2]
    if np.linalg.norm(new_gaze - old_gaze) <= 40:
        head_delay += 0.01

    head_start = float(eye_start + head_delay)
    head_end1 = float(head_start + head_travel_time)
    head_end2 = float(head_end1 + head_travel_time)

    # prepare return values
    fac = 1
    if delta_gaze <= 10:
        fac = 0.1
    head_keyframes.append([head_start, float(old_head_position[0]), float(old_head_position[1])])
    head_keyframes.append(
        [head_end1, float(old_head_position[0] + (new_head_angle_az - old_head_position[0]) * 0.8 * fac),
         float(old_head_position[1] + (new_head_angle_ele - old_head_position[1]) * 0.8 * fac)])
    head_keyframes.append([head_end2, float(new_head_angle_az), float(new_head_angle_ele * fac)])

    return gaze_key_frames, head_keyframes, previous_head_frame

def generate_neck_eye_curve_v2p3(time_arr, pos_arr, tags_arr):

    prev_gaze_angle = np.array([0.0, 0.0])
    prev_head_angle = np.array([0.0, 0.0])
    az_decomp = decomp.Heuristic_decomposition_azimuth()
    ele_decomp = decomp.Heuristic_decomposition_elevation()
    eye_v_decomp = decomp.Heuristic_eye_velocity()
    #     neck_velocity = decomp.
    eye_keyframes = []
    head_keyframes = []

    pos_arr = np.array(pos_arr)
    angles = decomp.gaze_vector_to_angle(pos_arr)
    for i in range(0, angles.shape[0]):
        # compute new angle
        goal_angle = angles[i]
        time = time_arr[i]
        # angle_difference = goal_angle - prev_gaze_angle
        new_eye_angle_az, new_head_angle_az = az_decomp.decompose(goal_angle[0], 0.5)
        new_eye_angle_ele, new_head_angle_ele = ele_decomp.decompose(goal_angle[1])
        if i == 0:
            eye_keyframes.append([[float(time_arr[0]), float(goal_angle[0]), float(goal_angle[1])]])
            head_keyframes.append([[float(time_arr[0]), float(new_head_angle_az), float(new_head_angle_ele)]])
            prev_gaze_angle = goal_angle.copy()
        else:
            eye, head, prev_head = handle_gaze_direction(prev_gaze_angle, head_keyframes[-1], goal_angle, time, tags_arr[i], az_decomp, ele_decomp, eye_v_decomp)
            if eye == None:
                continue
            head_keyframes[-1] = prev_head
            eye_keyframes.append(eye)
            if head != []:
                head_keyframes.append(head)
            # store previous head angle
            prev_gaze_angle = goal_angle

    # generate micro_saccades during fixation
    microsaccade_keyframes = []
    for i in range(0, len(eye_keyframes)):
        prev_saccade_frame = 0
        if i == 0:
            temp_out, prev_saccade_frame = handle_saccade(0, prev_saccade_frame,
                                                          eye_keyframes[i][0][0], saccade_factor=0.5)
        else:
            temp_out, prev_saccade_frame = handle_saccade(eye_keyframes[i - 1][-1][0], prev_saccade_frame,
                                                          eye_keyframes[i][0][0], saccade_factor=0.5)

        microsaccade_keyframes.append(temp_out)

    return eye_keyframes, head_keyframes, microsaccade_keyframes
# ek, hk = generate_neck_eye_curve_v2p2(out_time, out_pos, [])