import json
from typing import Dict, List
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
from Geometry_Util import rotation_angles_frome_positions, directions_from_rotation_angles
from Signal_processing_utils import intensity_from_signal
from scipy.interpolate import interp1d
from scipy.special import softmax
import librosa
class Dietic_Conversation_Gaze_Scene_Info:
    # speaker info
    speaker_position_world: np.array = np.zeros((3,)) 
    speaker_face_direction_local: np.array = np.zeros((3,))

    # transformations
    world_to_local: np.array = np.zeros((3, 3))
    local_to_world: np.array = np.zeros((3, 3))
    speaker_frame_pos: np.array = np.zeros((3, )) # position of the speaker head in world coord

    # the positions are normalized direction with resprect to the position of the head
    scene_object_id: List[str] = []

    # image based variables
    def __init__(self, scene_data_path):
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        self_dict = scene_data["self_pos"]
        self.speaker_position_world = np.array(self_dict["pos"])
        self.speaker_frame_pos = np.array(self_dict["pos"])
        self.speaker_face_direction_local = np.array(self_dict["calibration_dir_local"])
        v_ref_world = np.array(self_dict["calibration_dir_world"])
        v_ref_local = np.array(self_dict["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.speaker_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        temp_object_type, temp_object_pos, temp_object_interest = scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        self.scene_object_id = list(temp_object_pos.keys())
        self.object_type = []
        self.object_pos = []
        self.object_interest = []
        for i in range(0, len(self.scene_object_id)):
            self.object_type.append(temp_object_type[self.scene_object_id[i]])
            self.object_interest.append(temp_object_interest[self.scene_object_id[i]])
            self.object_pos.append(temp_object_pos[self.scene_object_id[i]])

        self.object_pos = np.array(self.object_pos)
        self.object_interest = np.array(self.object_interest)
        self.object_type = np.array(self.object_type)
        self.object_distance_to_listener = np.zeros(self.object_interest.shape)
        for i in range(0, self.object_type.shape[0]):
            if self.object_type[i] == 5:
                listener_direction_l = self.transform_world_to_local(self.object_pos[i])
                listener_direction_l = 1 / np.linalg.norm(listener_direction_l) * listener_direction_l
                break
        for i in range(0, self.object_type.shape[0]):
            obj_direction_l = self.transform_world_to_local(self.object_pos[i])
            obj_direction_l = 1 / np.linalg.norm(obj_direction_l) * obj_direction_l
            self.object_distance_to_listener[i] = obj_direction_l.dot(listener_direction_l)
        self.positions_world = self.object_pos
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100])):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[22, 10], [-22, 10], [-22, -10], [22, -10]]
        out_positions = []
        out_angles = []
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        return out_positions
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.speaker_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.speaker_position_world
        return p
    def get_conversation_partner_id(self):
        partners = []
        for i in range(0, self.object_type.shape[0]):
            if self.object_type[i] == 5:
                partners.append(i)
        return partners
class MultiPartyConversationalSceneInfo:
    speaker_position_world: np.array = np.zeros((3,3))
    speaker_face_direction_local: np.array = np.zeros((3,3))

    # transformations
    world_to_local: np.array = np.zeros((3, 3, 3))
    local_to_world: np.array = np.zeros((3, 3, 3))
    speaker_frame_pos: np.array = np.zeros((3, 3)) # position of the speaker head in world coord

    # the positions are normalized direction with resprect to the position of the head
    scene_object_id: List[str] = []
    def __init__(self, scene_data_path, audio, sample_rate):
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # initialize storage data structure
        self.positions_world = []
        self.object_type = []
        self.object_interest = []
        self.speakers_index = scene_data["speaker_indexes"]
        self.local_to_world = np.zeros((3, 3, 3))
        self.world_to_local = np.zeros((3, 3, 3))
        self.speaker_position_world = np.zeros((3, 3))
        self.speaker_frame_pos = np.zeros((3, 3))

        v_ref_local = np.zeros((3, 3))
        v_ref_world = np.zeros((3, 3))
        # get object pos, interestingness and type
        for i in range(0, len(scene_data["object_pos"].keys())):
            self.positions_world.append(scene_data["object_pos"][str(i)])
            self.object_type.append(scene_data["object_type"][str(i)])
            self.object_interest.append(scene_data["object_interestingness"][str(i)])
        self.positions_world = np.array(self.positions_world)
        self.object_type = np.array(self.object_type)
        self.object_interest = np.array(self.object_interest)
        # get speaker related info
        for i in range(len(self.speakers_index)):
            self.speaker_position_world[i] = self.positions_world[self.speakers_index[i]]
            self.speaker_frame_pos[i] = self.positions_world[self.speakers_index[i]]
            v_ref_world[i] = scene_data["calibration_global"][str(i)]
            v_ref_local[i] = scene_data["calibration_local"]
            self.speaker_face_direction_local = np.array(scene_data["calibration_local"])
            self.local_to_world[i] = self.rotation_matrix_from_vectors(v_ref_local[i],
                                                                    v_ref_world[i] - self.speaker_position_world[i])
            self.world_to_local[i] = np.linalg.inv(self.local_to_world[i])
        audio = [intensity_from_signal(f, int(sample_rate/50)) for f in audio]
        audios_expand = [np.expand_dims(f, axis=0) for f in audio]
        audios_expand = np.concatenate(audios_expand, axis=0)
        self.audio = audios_expand
        self.audio_t = np.arange(0, audios_expand.shape[1])/50
        self.intensity_interp = interp1d(self.audio_t, self.audio, bounds_error=False)
        return
    def get_pos_in_frame_i(self, i, object_id):
        return self.transform_world_to_local(self.positions_world[object_id], i)
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100])):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[22, 10], [-22, 10], [-22, -10], [22, -10]]
        out_positions = []
        out_angles = []
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        return out_positions
    def transform_world_to_local(self, pos_world, i):
        p = pos_world - self.speaker_position_world[i]
        return self.world_to_local[i] @ p
    def transform_local_to_world(self, pos_local, i):
        p = self.local_to_world[i] @ pos_local + self.speaker_position_world[i]
        return p
    def get_conversation_partner_id(self, self_id, t):
        speaker = np.argmax(self.intensity_interp(t))
        # if the person is not speaking, the speaker is the partner
        if speaker != self_id:
            return speaker
        t_counter = t
        # if the person is speaking, we assume the partner is the previous speaker
        while t_counter > 0:
            speaker = np.argmax(self.intensity_interp(t_counter))
            if speaker != self_id:
                return speaker
            t_counter = t_counter - 0.1
        for i in range(0, self.intensity_interp(t).shape[0]):
            if i != self_id:
                return i
class AgentInfo:
    # image based variables
    def __init__(self, scene_data_path, wonder = True):
        self.wonder = wonder
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # print(scene_data.keys())
        self_info = scene_data["self_pos"]
        # the position of the speaker, in world coordinate (constant in this version)
        self.self_position_world = np.array(self_info["pos"])
        self.speaker_frame_pos = np.array(self_info["pos"])
        # get info used to compute transformation matrix from world coordinate to face coordinate
        self.speaker_face_direction_local = np.array(self_info["calibration_dir_local"])
        v_ref_world = np.array(self_info["calibration_dir_world"])
        v_ref_local = np.array(self_info["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.self_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        # get info regarding other items in the scene
        temp_object_type, temp_object_pos, temp_object_interest = scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        temp_scene_object_ids = list(temp_object_pos.keys())
        # name of the passive objects of the scene
        self.object_pos = []
        self.object_interest = []
        self.object_id = []
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] < 5:
                self.object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
                self.object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
                self.object_id.append(temp_scene_object_ids[i])
        self.object_pos = np.array(self.object_pos)
        self.object_interest = np.array(self.object_interest)
        # name of the other active objec in the scene
        self.active_object_pos = []
        self.active_object_interest = []
        self.active_obejct_id = []
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] == 5:
                self.active_object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
                self.active_object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
                self.active_obejct_id.append(temp_scene_object_ids[i])  
        self.active_object_pos = np.array(self.active_object_pos)
        self.active_object_pos = np.array(self.active_object_pos)
        self.scene_data = scene_data
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1z, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100]), coordinate_space="local"):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[0, -10], [0, 8], [-10, -10], [10, -10], [10, 5], [-10, 5]]
        out_positions = []
        out_angles = []
        neutral_gaze_spot_local = self.get_active_object_position(0)
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        if coordinate_space == "local":
            return out_positions
        else:
            out = np.zeros(out_positions.shape)
            for i in range(0, out.shape[0]):
                out[i] = self.transform_local_to_world(out_positions[i])
            return out
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.self_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.self_position_world
        return p
    def get_object_positions(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.object_pos.shape)
                for i in range(0, self.object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.object_pos[i]
            elif coordinate_space == "local":
                out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
    def get_active_object_position(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.active_object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.active_object_pos.shape)
                for i in range(0, self.active_object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.active_object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.active_object_pos[id]
            elif coordinate_space == "local":
                return self.transform_world_to_local(self.active_object_pos[id])
    def get_all_positions(self, coordinate_space="local", index=-1):
        objs = self.get_object_positions(coordinate_space=coordinate_space)
        active_objs = self.get_active_object_position(coordinate_space=coordinate_space)
        wp = self.get_wondering_points(coordinate_space=coordinate_space, neutral_gaze_spot_local=active_objs[0])
        if self.wonder:
            possss = np.concatenate([objs, active_objs, wp], axis=0)
        else:
            possss = np.concatenate([objs, active_objs], axis=0)
        if index == -1:
            return possss
        else:
            return possss[index]
class TurnTakingData:
    def __init__(self, audio_path, audio_threshold=40):
        
        signal, sr = librosa.load(audio_path, sr=44100)
        intensity = intensity_from_signal(signal, int(sr/100))
        ts = np.arange(0, intensity.shape[0])/100
        activity = np.zeros(intensity.shape)
        # turn activity into 0 and 1
        for i in range(0, intensity.shape[0]):
            if intensity[i] - audio_threshold < 0:
                activity[i] = 0
            else:
                activity[i] = 1
        # segment the activity into segments of activity
        self.activity_list = []
        start = 0
        in_range = 0
        for i in range(0, activity.shape[0]):
            if i == activity.shape[0] - 1 and in_range == 1:
                self.activity_list.append([ts[start], ts[i]])
            elif in_range == 0 and activity[i] == 1:
                in_range = 1
                start = i
            elif in_range == 1 and activity[i] == 0:
                in_range = 0
                self.activity_list.append([ts[start], ts[i]])
    def plot(self):
        # Create an empty figure
        fig = plt.figure()

        # Loop through the list of tuples
        for speech in self.activity_list:
            start_time = speech[0]
            end_time = speech[1]
            # Plot a horizontal bar for each tuple
            plt.broken_barh([(start_time, end_time - start_time)], (0, 1), facecolors='blue')

        # Set the x-axis limit based on the maximum end time
        plt.xlim(-1, max(speech[1] for speech in self.activity_list) + 1)

        # Add x and y-axis labels
        plt.xlabel('Time')
        plt.ylabel('Task')

        # Display the plot
        plt.show()

class AgentInfo_time_varying:
    # image based variables
    def __init__(self, scene_data_path, wonder = True):
        self.wonder = wonder
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # print(scene_data.keys())
        self_info = scene_data["self_pos"]
        # the position of the speaker, in world coordinate (constant in this version)
        self.self_position_world = np.array(self_info["pos"])
        self.speaker_frame_pos = np.array(self_info["pos"])
        # get info used to compute transformation matrix from world coordinate to face coordinate
        self.speaker_face_direction_local = np.array(self_info["calibration_dir_local"])
        v_ref_world = np.array(self_info["calibration_dir_world"])
        v_ref_local = np.array(self_info["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.self_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        # get info regarding other items in the scene
        temp_object_type, temp_object_pos, temp_object_interest = scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        temp_scene_object_ids = list(temp_object_pos.keys())
        # name of the passive objects of the scene
        self.object_pos = []
        self.object_interest = []
        self.object_id = []
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] < 5:
                self.object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
                self.object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
                self.object_id.append(temp_scene_object_ids[i])
        self.object_pos = np.array(self.object_pos)
        for i in range(0, len(self.object_interest)):
            self.object_interest[i] = np.array(self.object_interest[i])        

        # name of the other active objec in the scene
        self.active_object_pos = []
        self.active_object_interest = []
        self.active_obejct_id = []
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] == 5:
                self.active_object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
                self.active_object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
                self.active_obejct_id.append(temp_scene_object_ids[i])  
        self.active_object_pos = np.array(self.active_object_pos)
        self.active_object_pos = np.array(self.active_object_pos)
        self.scene_data = scene_data
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1z, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100]), coordinate_space="local"):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[0, -10], [0, 8], [-10, -10], [10, -10], [10, 5], [-10, 5]]
        out_positions = []
        out_angles = []
        neutral_gaze_spot_local = self.get_active_object_position(0)
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        if coordinate_space == "local":
            return out_positions
        else:
            out = np.zeros(out_positions.shape)
            for i in range(0, out.shape[0]):
                out[i] = self.transform_local_to_world(out_positions[i])
            return out
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.self_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.self_position_world
        return p
    def get_object_positions(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.object_pos.shape)
                for i in range(0, self.object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.object_pos[i]
            elif coordinate_space == "local":
                out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
    def get_active_object_position(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.active_object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.active_object_pos.shape)
                for i in range(0, self.active_object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.active_object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.active_object_pos[id]
            elif coordinate_space == "local":
                return self.transform_world_to_local(self.active_object_pos[id])
    def get_all_positions(self, coordinate_space="local", index=-1):
        objs = self.get_object_positions(coordinate_space=coordinate_space)
        active_objs = self.get_active_object_position(coordinate_space=coordinate_space)
        wp = self.get_wondering_points(coordinate_space=coordinate_space, neutral_gaze_spot_local=active_objs[0])
        if self.wonder:
            possss = np.concatenate([objs, active_objs, wp], axis=0)
        else:
            possss = np.concatenate([objs, active_objs], axis=0)
        if index == -1:
            return possss
        else:
            return possss[index]
    def get_interet(self, object_id, t):
        interest_arr = self.object_interest[object_id]
        if interest_arr.shape[0] == 1:
            return interest_arr[0][1]
        else:
            for i in range(1, interest_arr.shape[0]):
                if interest_arr[i][0] > t:
                    return interest_arr[i-1][1]


class AgentInfo_final:
    # image based variables
    def __init__(self, scene_data_path, wonder = True):
        self.wonder = wonder
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # print(scene_data.keys())
        self_info = scene_data["self_pos"]
        # get camera position
        try:
            self.camera_pos = np.array(scene_data["cam_pos"])
        except:
            self.camera_pos = np.array(self_info["calibration_dir_world"])
        # the position of the speaker, in world coordinate (constant in this version)
        self.self_position_world = np.array(self_info["pos"])
        self.speaker_frame_pos = np.array(self_info["pos"])
        # get info used to compute transformation matrix from world coordinate to face coordinate
        self.speaker_face_direction_local = np.array(self_info["calibration_dir_local"])
        v_ref_world = np.array(self_info["calibration_dir_world"])
        v_ref_local = np.array(self_info["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.self_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        # get info regarding other items in the scene
        temp_object_type, temp_object_pos, temp_object_interest = scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        temp_scene_object_ids = list(temp_object_pos.keys())
        # name of the passive objects of the scene
        self.object_pos = []
        self.object_interest = []
        self.object_id = []
        for i in range(0, len(temp_scene_object_ids)):
            self.object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
            self.object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
            self.object_id.append(temp_scene_object_ids[i])
        self.object_pos = np.array(self.object_pos)
        for i in range(0, len(self.object_interest)):
            self.object_interest[i] = np.array(self.object_interest[i])
        self.active_object_id = 0
        self.other_speaker_id = 0
        # get the id of the other active objec in the scene
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] == 5:
                self.other_speaker_id = i
                self.active_object_id = i
        self.scene_data = scene_data
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1z, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100]), coordinate_space="local"):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[0, -10], [0, 8], [-5, -5], [5, -5], [5, 5], [-5, 5]]
        # wondering_angles = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
        # print(wondering_angles)
        out_positions = []
        out_angles = []
        neutral_gaze_spot_local = self.get_active_object_position()
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        if coordinate_space == "local":
            return out_positions
        else:
            out = np.zeros(out_positions.shape)
            for i in range(0, out.shape[0]):
                out[i] = self.transform_local_to_world(out_positions[i])
            return out
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.self_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.self_position_world
        return p
    def get_object_positions(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.object_pos.shape)
                for i in range(0, self.object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.object_pos[id]
            elif coordinate_space == "local":
                out = self.transform_world_to_local(self.object_pos[id])
                return out
    def get_active_object_position(self, coordinate_space="local"):
        id = self.active_object_id
        return self.get_object_positions(id, coordinate_space=coordinate_space)
    def get_all_positions(self, coordinate_space="local"):
        objs = self.get_object_positions(coordinate_space=coordinate_space)
        active_objs = self.get_active_object_position()
        wp = self.get_wondering_points(coordinate_space=coordinate_space, neutral_gaze_spot_local=active_objs)
        if self.wonder:
            possss = np.concatenate([objs, wp], axis=0)
        else:
            possss = np.concatenate([objs], axis=0)
        return possss
    def get_interest(self, object_id, t):
        interest_arr = self.object_interest[object_id]
        # print(interest_arr)
        if interest_arr.shape[0] == 1:
            return interest_arr[0][1]
        else:
            for i in range(1, interest_arr.shape[0]):
                if interest_arr[i][0] > t:
                    return interest_arr[i-1][1]
        return interest_arr[-1][1]
    def get_camera_pos(self, coordinate_space="local"):
        if coordinate_space == "local":
            return self.transform_world_to_local(self.camera_pos)
        else:
            return self.camera_pos
class AgentInfo_final_multiparty:
    # image based variables
    def __init__(self, scene_data_path, wonder = True):
        self.wonder = wonder
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # print(scene_data.keys())
        self_info = scene_data["self_pos"]
        # get camera position
        try:
            self.camera_pos = np.array(scene_data["cam_pos"])
        except:
            self.camera_pos = np.array(self_info["calibration_dir_world"])
        # the position of the speaker, in world coordinate (constant in this version)
        self.self_position_world = np.array(self_info["pos"])
        self.speaker_frame_pos = np.array(self_info["pos"])
        # get info used to compute transformation matrix from world coordinate to face coordinate
        self.speaker_face_direction_local = np.array(self_info["calibration_dir_local"])
        v_ref_world = np.array(self_info["calibration_dir_world"])
        v_ref_local = np.array(self_info["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.self_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        # get info regarding other items in the scene
        temp_object_type, temp_object_pos, temp_object_interest = scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        temp_speaker_id = scene_data["speaker_id"]
        temp_scene_object_ids = list(temp_object_pos.keys())
        # name of the passive objects of the scene
        self.object_pos = []
        self.object_interest = []
        self.object_id = []
        self.speaker_id = []
        for i in range(0, len(temp_scene_object_ids)):
            self.object_interest.append(temp_object_interest[temp_scene_object_ids[i]])
            self.object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
            self.object_id.append(temp_scene_object_ids[i])
            self.speaker_id.append(temp_speaker_id[temp_scene_object_ids[i]])
        self.object_pos = np.array(self.object_pos)
        for i in range(0, len(self.object_interest)):
            self.object_interest[i] = np.array(self.object_interest[i])
        self.active_object_id = []
        self.other_speaker_id = []
        # get the id of the other active objec in the scene
        for i in range(0, len(temp_scene_object_ids)):
            if temp_object_type[temp_scene_object_ids[i]] == 5:
                self.other_speaker_id.append(i)
                self.active_object_id.append(i)
        self.active_object_id = int(np.minimum(*self.active_object_id))
        self.scene_data = scene_data
    def change_speaker(self, speaker_id):
        # find the index of whcih the speaker_id matches the parameter
        print(speaker_id, self.speaker_id)
        for i in range(0, len(self.speaker_id)):
            if self.speaker_id[i][1] == speaker_id:
                self.active_object_id = i
                return True
        print("speaker not found")
        return False
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1z, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def get_wondering_points(self, neutral_gaze_spot_local=np.array([0, 0, 100]), coordinate_space="local"):
        """
        get all the looking positions in local frame
        :param neutral_gaze_spot_local: the default gaze position
        :return: a [6, 3] array of all the positions to wonder
        """
        wondering_angles = [[0, -10], [0, 8], [-5, -5], [5, -5], [5, 5], [-5, 5]]
        # wondering_angles = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
        # print(wondering_angles)
        out_positions = []
        out_angles = []
        neutral_gaze_spot_local = self.get_active_object_position()
        neutral_gaze_angle = rotation_angles_frome_positions(neutral_gaze_spot_local)
        for angle in wondering_angles:
            new_angle = np.zeros((1, 2))
            new_angle[0, 0] = neutral_gaze_angle[0] + angle[0]
            new_angle[0, 1] = neutral_gaze_angle[1] + angle[1]
            out_angles.append(new_angle)
        out_angles = np.concatenate(out_angles, axis=0)
        out_positions = directions_from_rotation_angles(out_angles, np.linalg.norm(neutral_gaze_spot_local))
        if coordinate_space == "local":
            return out_positions
        else:
            out = np.zeros(out_positions.shape)
            for i in range(0, out.shape[0]):
                out[i] = self.transform_local_to_world(out_positions[i])
            return out
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.self_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.self_position_world
        return p
    def get_object_positions(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.object_pos.shape)
                for i in range(0, self.object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.object_pos[id]
            elif coordinate_space == "local":
                out = self.transform_world_to_local(self.object_pos[id])
                return out
    def get_active_object_position(self, coordinate_space="local"):
        id = self.active_object_id
        return self.get_object_positions(id, coordinate_space=coordinate_space)
    def get_all_positions(self, coordinate_space="local"):
        objs = self.get_object_positions(coordinate_space=coordinate_space)
        active_objs = self.get_active_object_position()
        wp = self.get_wondering_points(coordinate_space=coordinate_space, neutral_gaze_spot_local=active_objs)
        if self.wonder:
            possss = np.concatenate([objs, wp], axis=0)
        else:
            possss = np.concatenate([objs], axis=0)
        return possss
    def get_interest(self, object_id, t):
        interest_arr = self.object_interest[object_id]
        # print(interest_arr)
        if interest_arr.shape[0] == 1:
            return interest_arr[0][1]
        else:
            for i in range(1, interest_arr.shape[0]):
                if interest_arr[i][0] > t:
                    return interest_arr[i-1][1]
        return interest_arr[-1][1]
    def get_camera_pos(self, coordinate_space="local"):
        if coordinate_space == "local":
            return self.transform_world_to_local(self.camera_pos)
        else:
            return self.camera_pos

if __name__ == "__main__":
    a = AgentInfo("/Users/evanpan/Documents/GitHub/Gaze_project/data/look_at_points/simplest_scene2_less_items.json")
    print(a.get_active_object_position(0, coordinate_space="global"))
    