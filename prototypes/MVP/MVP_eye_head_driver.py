import numpy as np
from prototypes.Jin2019.EyeHeadDecomposition import Heuristic_decomposition_azimuth, Heuristic_decomposition_elevation, Heuristic_eye_velocity, Heuristic_eye_velocity_simple
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from Geometry_Util import rotation_angles_frome_positions, directions_from_rotation_angles
from Speech_Data_util import Sentence_word_phone_parser
from Curve_utils import ImpluseSpineCurveModel
from matplotlib import pyplot as plt
class HeuristicGazeMotionGenerator():
    def __init__(self, scene:Dietic_Conversation_Gaze_Scene_Info, sementic_script: Sentence_word_phone_parser):
        self.azimuth_decomp = Heuristic_decomposition_azimuth()
        self.elevation_decomp = Heuristic_decomposition_elevation()
        self.velocity_decomp = Heuristic_eye_velocity_simple()
        self.scene = scene
        self.head_x = ImpluseSpineCurveModel()
        self.head_y = ImpluseSpineCurveModel()
        self.gaze_x = ImpluseSpineCurveModel()
        self.gaze_y = ImpluseSpineCurveModel()
        self.sementic_script = sementic_script

    def handle_saccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05, avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 1  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def handle_gaze_direction(self, old_gaze, previous_head_frame, new_gaze, new_time,
                              parameters, az_decomp, ele_decomp, eye_v_decomp):
        head_keyframes = []
        gaze_key_frames = []

        reaction_time_factor = parameters[0]
        head_contribution_factor = parameters[1]
        head_delay_factor = parameters[2]
        head_speed_factor = parameters[3]
        # in case I give the head an unwanted jitter
        if np.linalg.norm(old_gaze - new_gaze) == 0:
            return None, None, None

            # stupidly fast reaction time is 0.1 second, on average it is betweem 0.2-0.3s
        reaction_time = 0.3 - 0.2 * reaction_time_factor
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


        ##############################################################
        ######################### BANDAID ############################
        ##############################################################

        # dealing with bad cases of head movements
        old_head_position = np.array([previous_head_frame[-1][1], previous_head_frame[-1][2]])

        ##############################################################
        ########################### NECK #############################
        ##############################################################

        # compute head amplitude
        new_eye_angle_az, new_head_angle_az = az_decomp.decompose(new_gaze[0], head_contribution_factor)
        new_eye_angle_ele, new_head_angle_ele = ele_decomp.decompose(new_gaze[1])

        # compute head timing (velocity)
        new_head_angle = np.array([new_head_angle_az, new_head_angle_ele])
        delta_head = np.linalg.norm(new_head_angle - old_head_position)
        head_speed = min((2 + (head_speed_factor) * 4) * delta_head + 20, 300)
        head_travel_time = delta_head / head_speed

        # compute head movement delay (could be very slow could be fast)

        head_delay = 0 + 0.05 * head_delay_factor
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

        # preparing output structure for gaze
        gaze_key_frames.append([eye_start, float(old_gaze[0]), float(old_gaze[1])])
        gaze_key_frames.append([eye_end, float(new_gaze[0]), float(new_gaze[1])])

        return gaze_key_frames, head_keyframes, previous_head_frame
    def handle_gaze_shift_legacy(self, old_gaze, old_head, new_gaze, new_head, interval, parameters):
        # unpack parameters
        reaction_time_factor = parameters[0]
        head_contribution_factor = parameters[1]
        head_delay_factor = parameters[2]
        head_speed_factor = parameters[3]

        # do nothing if the new and old angle is the same
        if np.linalg.norm(old_gaze - new_gaze) == 0:
            return None, None, None

        # reaction time is usually around 0.1 to 0.3 seconds
        reaction_time = 0.3 - 0.2 * reaction_time_factor
        reaction_time += np.random.normal(0, 0.005)  # as per literature

        ##############################################################
        ######################### EYESSSS ############################
        ##############################################################
        # compute timing (velocity) for the eye based on heuristics.
        delta_gaze = np.linalg.norm(old_gaze - new_gaze)
        gaze_speed = self.velocity_decomp.decompose(delta_gaze)
        eye_travel_time = delta_gaze / gaze_speed

        # compute actual timings of the keyframes
        eye_transition_start = float(interval[0] + reaction_time)
        eye_transition_end = float(eye_transition_start + eye_travel_time)
        eye_hold_end = float(interval[1])

        gaze_t = [eye_transition_start, eye_transition_end, eye_hold_end]
        gaze_x = [float(old_gaze[0]), float(new_gaze[0]), float(new_gaze[0])]
        gaze_y = [float(old_gaze[1]), float(new_gaze[1]), float(new_gaze[1])]

        stable_gaze = [float(eye_transition_end), float(eye_hold_end)]
        ##############################################################
        ######################### NECKKKK ############################
        ##############################################################
        # compute head amplitude
        new_head_angle_az = new_head[0]
        new_head_angle_ele = new_head[1]

        # compute head timing (velocity)
        new_head_angle = np.array([new_head_angle_az, new_head_angle_ele])
        delta_head = np.linalg.norm(new_head_angle - old_head)
        head_speed = min((2 + (head_speed_factor) * 4) * delta_head + 20, 300)
        head_travel_time = delta_head / head_speed

        # compute head movement delay (could be very slow could be fast)

        head_delay = 0 + 0.05 * head_delay_factor
        if np.linalg.norm(new_gaze - old_gaze) <= 40:
            head_delay += 0.01

        head_start = float(eye_transition_start + head_delay)
        head_end1 = float(head_start + head_travel_time)
        head_end2 = float(head_end1 + head_travel_time)

        # prepare return values
        fac = 1
        if delta_gaze <= 10:
            fac = 0.1
        if interval[1] > head_end2:
            neck_t = [head_start, head_end1, head_end2, interval[1]]
            neck_x = [float(old_head[0]), float(old_head[0] + (new_head_angle_az - old_head[0]) * 0.8 * fac), float(new_head_angle_az), float(new_head_angle_az)]
            neck_y = [float(old_head[1]), float(old_head[1] + (new_head_angle_ele - old_head[1]) * 0.8 * fac), float(new_head_angle_ele), float(new_head_angle_ele)]
        else:
            neck_t = [head_start, head_end1, head_end2]
            neck_x = [float(old_head[0]), float(old_head[0] + (new_head_angle_az - old_head[0]) * 0.8 * fac),
                      float(new_head_angle_az), ]
            neck_y = [float(old_head[1]), float(old_head[1] + (new_head_angle_ele - old_head[1]) * 0.8 * fac),
                      float(new_head_angle_ele)]
        self.update_ImpulseCurveModel(gaze_t, gaze_x, gaze_y, neck_t, neck_x, neck_y)

        return new_gaze, new_head_angle, stable_gaze
    def generate_neck_eye_curve(self, time_arr, pos_arr):

        head_delay_factor = 0.5
        head_speed_factor = 0.1
        gaze_intervals_time = []
        gaze_intervals_pos = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i]-pos_arr[i-1]) != 0:
                gaze_intervals_time.append([start, time_arr[-2]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i]-pos_arr[i-1]) == 0:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i]-pos_arr[i-1]) != 0:
                gaze_intervals_time.append([start, time_arr[i]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[i-1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i]-pos_arr[i-1]) == 0:
                pass
            else:
                print("I wanna eat korean fried chicken")

        raw_gaze_key_frames = []
        raw_neck_key_frames = []

        angles = rotation_angles_frome_positions(np.concatenate(gaze_intervals_pos, axis=0))
        old_gaze = angles[0]
        for i in range(0, len(gaze_intervals_pos)):
            angle = angles[i]
            time = gaze_intervals_time[i]
            raw_gaze_key_frames.append([[time[0], angle[0], angle[1]]])
        head_travel_times = []
        for i in range(0, len(raw_gaze_key_frames)):
            raw_neck_key_frames_i = []
            t = raw_gaze_key_frames[i][0][0]
            angle = np.array([raw_gaze_key_frames[i][0][1], raw_gaze_key_frames[i][0][2]])
            # get speed component of gaze
            if i > 0:
                prev_angle = np.array([raw_gaze_key_frames[i-1][0][1], raw_gaze_key_frames[i-1][0][2]])
            else:
                prev_angle = rotation_angles_frome_positions(np.expand_dims(self.scene.speaker_face_direction, axis=0))[0]
            delta_gaze = np.linalg.norm(prev_angle - angles[i])
            gaze_speed = self.velocity_decomp.decompose(delta_gaze)
            # this time is kind of important
            eye_travel_time = delta_gaze / gaze_speed
            # get the neck angle component

            interval_i = gaze_intervals_time[i]
            neck_contribution = (interval_i[1] - interval_i[0]) / 4.0  # the longer, the more neck contribution
            neck_contribution = min(neck_contribution, 1)
            # now we've gotten the head angles
            new_eye_angle_az, new_head_angle_az = self.azimuth_decomp.decompose(angle[0], neck_contribution)
            new_eye_angle_ele, new_head_angle_ele = self.elevation_decomp.decompose(angle[1], neck_contribution)
            if i > 0:
                old_head = np.array([raw_neck_key_frames[-1][0][1], raw_neck_key_frames[-1][0][0]])
            else:
                old_head = rotation_angles_frome_positions(np.expand_dims(self.scene.speaker_face_direction, axis=0))[0]

            new_head_angle = np.array([new_head_angle_az, new_head_angle_ele])
            delta_head = np.linalg.norm(new_head_angle - old_head)
            head_speed = min((2 + (head_speed_factor) * 4) * delta_head + 20, 300)
            head_travel_time = delta_head / head_speed
            head_travel_times.append(head_travel_time)
            t_look = t - eye_travel_time + 0.05 * head_delay_factor + head_travel_time
            

            # prevent the neck from moving too much
            too_clinch = False
            for j in range(len(raw_neck_key_frames)-1, -1, -1):
                if raw_neck_key_frames[j][0][0] >= t_look:
                    too_clinch = True
                    break
            if too_clinch:
                continue
            else:
                raw_neck_key_frames_i.append([t_look, new_head_angle_az, new_head_angle_ele])
                raw_neck_key_frames.append(raw_neck_key_frames_i)
        for i in range(0, len(raw_gaze_key_frames)):
            if i == 0:
                zero_look = rotation_angles_frome_positions(np.expand_dims(self.scene.speaker_face_direction, axis=0))[0]
                delta_gaze = np.linalg.norm(zero_look - angles[i])
                gaze_speed = self.velocity_decomp.decompose(delta_gaze)
                eye_travel_time = delta_gaze / gaze_speed
                raw_gaze_key_frames[i] = [[raw_gaze_key_frames[i][0][0] - eye_travel_time,
                                           zero_look[0], zero_look[1]]] + raw_gaze_key_frames[i]
            else:
                delta_gaze = np.linalg.norm(angles[i - 1] - angles[i])
                gaze_speed = self.velocity_decomp.decompose(delta_gaze)
                eye_travel_time = delta_gaze / gaze_speed
                # prevent anticipation frame from being added if the current gaze shift timing is too close to the previous
                too_close = False
                for j in range(i-1, -1, -1):
                    if raw_gaze_key_frames[i][0][0] - eye_travel_time < raw_gaze_key_frames[j][-1][0]:
                        too_close = True
                        break
                if too_close:
                    continue
                if (raw_gaze_key_frames[i][0][0] - raw_gaze_key_frames[i-1][-1][0] >= eye_travel_time):
                    # if the current gaze angle transition warrents a transition
                    raw_gaze_key_frames[i] = [[raw_gaze_key_frames[i][0][0] - eye_travel_time,
                                               angles[i-1][0], angles[i-1][1]]] + raw_gaze_key_frames[i]
                    raw_gaze_key_frames[i-1] = raw_gaze_key_frames[i-1] + [[float(raw_gaze_key_frames[i][0][0] - eye_travel_time),
                                               angles[i-1][0], angles[i-1][1]]]
                else:
                    raw_gaze_key_frames[i - 1] = raw_gaze_key_frames[i - 1] + [
                        [float(raw_gaze_key_frames[i][0][0] - eye_travel_time),
                         angles[i][0], angles[i][1]]]

            if i == len(raw_gaze_key_frames)-1:
                raw_gaze_key_frames[i] = raw_gaze_key_frames[i] + [[float(raw_gaze_key_frames[i][-1][0] + 10),
                                           angles[i][0], angles[i][1]]]
        for i in range(0, len(raw_neck_key_frames)):
            angle = np.array([raw_neck_key_frames[i][0][1], raw_neck_key_frames[i][0][2]])
            if i == 0:
                zero_look = rotation_angles_frome_positions(np.expand_dims(self.scene.speaker_face_direction, axis=0))[
                    0]
                delta_head = np.linalg.norm(new_head_angle - zero_look)
                head_speed = min((2 + (head_speed_factor) * 4) * delta_head + 20, 300)
                head_travel_time = delta_head / head_speed
                raw_neck_key_frames[i] = [[raw_neck_key_frames[i][0][0] - head_travel_time,
                                           zero_look[0], zero_look[1]]] + raw_neck_key_frames[i]
            else:
                prev_angle = np.array([raw_neck_key_frames[i-1][-1][1], raw_neck_key_frames[i-1][-1][2]])
                delta_head = np.linalg.norm(angle - prev_angle)
                head_speed = min((2 + (head_speed_factor) * 4) * delta_head + 20, 300)
                head_travel_time = delta_head / head_speed
                # prevent anticipation frame from being added if the current gaze shift timing is too close to the previous
                too_close = False
                for j in range(i - 1, -1, -1):
                    if raw_neck_key_frames[i][0][0] - head_travel_time < raw_neck_key_frames[j][-1][0]:
                        too_close = True
                        break
                if too_close:
                    continue
                if (raw_neck_key_frames[i][0][0] - raw_neck_key_frames[i - 1][-1][0] >= head_travel_time):
                    # if the current gaze angle transition warrents a transition
                    raw_neck_key_frames[i] = [[raw_neck_key_frames[i][0][0] - head_travel_time,
                                               prev_angle[0], prev_angle[1]]] + raw_neck_key_frames[i]
                    raw_neck_key_frames[i-1] = raw_neck_key_frames[i-1] + [
                        [float(raw_neck_key_frames[i][0][0] - head_travel_time),
                         prev_angle[0], prev_angle[1]]]
            if i == len(raw_gaze_key_frames) - 1:
                raw_gaze_key_frames[i] = raw_gaze_key_frames[i] + [[float(raw_gaze_key_frames[i][-1][0] + 10),
                                                                    angles[i][0], angles[i][1]]]

        for i in range(0, len(raw_neck_key_frames)):
            print(raw_neck_key_frames[i])


        for i in range(0, len(raw_gaze_key_frames)):
            # obtain rotation from all they keyframes
            for j in range(0, len(raw_gaze_key_frames[i])):
                t = raw_gaze_key_frames[i][j][0]
                eye_rot = []
                eye_rot.append([raw_gaze_key_frames[i][j][1], raw_gaze_key_frames[i][j][2]])
                eye_rot = np.array(eye_rot)
                eye_pos_local = directions_from_rotation_angles(eye_rot, 100)[0]
                raw_gaze_key_frames[i][j] = [t, float(eye_pos_local[0]), float(eye_pos_local[1]), float(eye_pos_local[2])]
        for i in range(0, len(raw_gaze_key_frames)):
            for j in range(0, len(raw_gaze_key_frames[i])):
                for k in range(0, len(raw_gaze_key_frames[i][j])):
                    raw_gaze_key_frames[i][j][k] = float(raw_gaze_key_frames[i][j][k])
        for i in range(0, len(raw_neck_key_frames)):
            for j in range(0, len(raw_neck_key_frames[i])):
                for k in range(0, len(raw_neck_key_frames[i][j])):
                    raw_neck_key_frames[i][j][k] = float(raw_neck_key_frames[i][j][k])
        return raw_gaze_key_frames, raw_neck_key_frames, []
    def generate_neck_eye_curve_legacy(self, time_arr, pos_arr):
        return self.simple_generate_neck_eye_curve(time_arr, pos_arr)
        gaze_intervals = []
        gaze_interval_arr = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i]-pos_arr[i-1]) != 0:
                gaze_intervals.append([start, time_arr[-2]])
                gaze_interval_arr.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals.append([time_arr[-1], time_arr[-1] + 10])
                gaze_interval_arr.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i]-pos_arr[i-1]) == 0:
                gaze_intervals.append([start, time_arr[-1]])
                gaze_interval_arr.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i]-pos_arr[i-1]) != 0:
                gaze_intervals.append([start, time_arr[i]])
                gaze_interval_arr.append(np.expand_dims(pos_arr[i-1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i]-pos_arr[i-1]) == 0:
                pass
            else:
                print("I wanna eat korean fried chicken")



        prev_gaze_angle = np.array([0.0, 0.0])
        prev_head_angle = np.array([0.0, 0.0])
        prev_saccade_frame = 0
        microsaccade_keyframes = []
        az_decomp = self.azimuth_decomp
        ele_decomp = self.elevation_decomp
        eye_v_decomp = self.velocity_decomp

        # get the angles from the positions for better break down
        angles = rotation_angles_frome_positions(np.concatenate(gaze_interval_arr, axis=0))
        for i in range(0, angles.shape[0]):
            # get the goal angle and everything
            goal_angle = angles[i]
            interval_i = gaze_intervals[i]

            # neck_contribution is dependent on the angle and how long the gaze will stay there
            neck_contribution = (interval_i[1] - interval_i[0]) / 1.0 # the longer, the more neck contribution
            neck_contribution = min(neck_contribution, 1)
            new_eye_angle_az, new_head_angle_az = az_decomp.decompose(goal_angle[0], neck_contribution)
            new_eye_angle_ele, new_head_angle_ele = ele_decomp.decompose(goal_angle[1])
            new_gaze_angle = goal_angle
            new_neck_angle = np.array([new_head_angle_az, new_head_angle_ele])
            # [reaction speed, head contribution, head reactivity, head velocity]
            gaze_shift_parameter = [0.5, 0.5, 1, 0]
            temp_prev_gaze_angle, temp_prev_head_angle, temp_gaze_fixation = self.handle_gaze_shift(prev_gaze_angle, prev_head_angle, new_gaze_angle, new_neck_angle, interval_i, gaze_shift_parameter)
            if temp_prev_gaze_angle is None:
                continue
            else:
                prev_gaze_angle = temp_prev_gaze_angle
                prev_head_angle = temp_prev_head_angle
                gaze_fixation = temp_gaze_fixation
            temp_out, prev_saccade_frame = self.handle_saccade(gaze_fixation[0], prev_saccade_frame,
                                gaze_fixation[1], saccade_factor=0.5)
            microsaccade_keyframes.append(temp_out)
        # compute curves
        eye_keyframes, head_keyframes = self.get_curve_ImpulseCurveModel()
        head_t = []
        head_x = []
        for i in range(0, len(head_keyframes)):
            for j in range(0, len(head_keyframes[i])):
                head_t.append(head_keyframes[i][j][0])
                head_x.append(head_keyframes[i][j][1])
        plt.plot(head_t, head_x)
        plt.show()

        for i in range(0, len(eye_keyframes)):
            # obtain rotation from all they keyframes
            for j in range(0, len(eye_keyframes[i])):
                t = eye_keyframes[i][j][0]
                eye_rot = []
                eye_rot.append([eye_keyframes[i][j][1], eye_keyframes[i][j][2]])
                eye_rot = np.array(eye_rot)
                eye_pos_local = directions_from_rotation_angles(eye_rot, 100)[0]
                eye_keyframes[i][j] = [t, float(eye_pos_local[0]), float(eye_pos_local[1]), float(eye_pos_local[2])]
        return eye_keyframes, head_keyframes, microsaccade_keyframes
    def update_ImpulseCurveModel(self, gaze_t, gaze_x, gaze_y, neck_t, neck_x, neck_y):
        neck_x_pts = [neck_t,neck_x]
        if len(neck_x_pts[0]) > 0:
            self.head_x.add_to_interval_tier((neck_x_pts), 0)
        neck_y_pts = [neck_t,neck_y]
        if len(neck_y_pts[0]) > 0:
            self.head_y.add_to_interval_tier((neck_y_pts), 0)
        gaze_x_pts = [[],[]]
        for i in range(0, len(gaze_t)):
            gaze_x_pts[0].append(gaze_t[i])
            gaze_x_pts[1].append(-gaze_x[i])
        if len(gaze_t) > 0:
            self.gaze_x.add_to_interval_tier((gaze_x_pts), 0)
        gaze_y_pts = [[],[]]
        for i in range(0, len(gaze_t)):
            gaze_y_pts[0].append(gaze_t[i])
            gaze_y_pts[1].append(-gaze_y[i])
        if len(gaze_t) > 0:
            self.gaze_y.add_to_interval_tier((gaze_y_pts), 0)
    def update_ImpulseCurveModel_with_interavls(self, gaze_t, gaze_x, gaze_y, neck_t, neck_x, neck_y):
        neck_x_pts = [neck_t,neck_x]
        if len(neck_x_pts[0]) > 0:
            self.head_x.add_to_interval_tier((neck_x_pts), 0)
        neck_y_pts = [neck_t,neck_y]
        if len(neck_y_pts[0]) > 0:
            self.head_y.add_to_interval_tier((neck_y_pts), 0)
        gaze_x_pts = [[],[]]
        for i in range(0, len(gaze_t)):
            gaze_x_pts[0].append(gaze_t[i])
            gaze_x_pts[1].append(-gaze_x[i])
        if len(gaze_t) > 0:
            self.gaze_x.add_to_interval_tier((gaze_x_pts), 0)
        gaze_y_pts = [[],[]]
        for i in range(0, len(gaze_t)):
            gaze_y_pts[0].append(gaze_t[i])
            gaze_y_pts[1].append(-gaze_y[i])
        if len(gaze_t) > 0:
            self.gaze_y.add_to_interval_tier((gaze_y_pts), 0)
    def get_curve_ImpulseCurveModel(self):
        self.head_x.fill_holes()
        self.head_y.fill_holes()
        self.gaze_x.fill_holes()
        self.gaze_y.fill_holes()

        self.head_x.recompute_all_points()
        self.head_y.recompute_all_points()
        self.gaze_x.recompute_all_points()
        self.gaze_y.recompute_all_points()
        EIH_x = ImpluseSpineCurveModel()
        eih_xt, eih_x = EIH_x.sum_with_other_tiers([self.head_x, self.gaze_x])
        EIH_y = ImpluseSpineCurveModel()
        eih_yt, eih_y = EIH_y.sum_with_other_tiers([self.head_y, self.gaze_y])

        # plt.plot(eih_yt, eih_y)
        # for i in self.gaze_y.control_points:
        #     print(self.gaze_y.control_points[i])
        # plt.plot(self.gaze_y.cached_xt, self.gaze_y.cached_x)
        # plt.show()
        head_kf = []
        for i in range(0, len(self.head_x.cached_x)):
            head_kf.append([self.head_x.cached_xt[i], self.head_x.cached_x[i], self.head_y.cached_x[i]])
        head_kf = [head_kf]

        eye_kf = []
        for i in range(0, len(eih_yt)):
            eye_kf.append([eih_yt[i], -eih_x[i], -eih_y[i]])
        eye_kf = [eye_kf]
        return eye_kf, head_kf


if __name__ == "__main__":
    pass