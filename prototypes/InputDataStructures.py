import json
from typing import Dict, List
import numpy as np

class Gaze_Scene_Info:
    # speaker info
    speaker_position: np.array = np.zeros((3, ))
    speaker_face_direction: np.array = np.zeros((3,))

    # transformations
    world_to_local: np.array = np.zeros((3, 3))
    local_to_world: np.array = np.zeros((3, 3))
    speaker_frame_pos: np.array = np.zeros((3, ))

    # the positions are normalized direction with resprect to the position of the head
    scene_object_id: List[str] = []
    object_pos: Dict[str, np.array]
    object_type: Dict[str, int]
    object_interest: Dict[str, float]

    # image based variables
    def __init__(self, scene_data_path):
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        print(scene_data.keys())
        self_dict = scene_data["self_pos"]
        print(self_dict.keys())
        self.speaker_position = np.array(self_dict["pos"])
        self.speaker_frame_pos = np.array(self_dict["pos"])
        v_ref_world = np.array(self_dict["calibration_dir_world"])
        v_ref_local = np.array(self_dict["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.speaker_position)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        self.object_type, self.object_pos, self.object_interest =  scene_data["object_type"], scene_data["object_pos"], scene_data["object_interestingness"]
        for key in self.object_pos:
            self.object_pos[key] = np.array(self.object_pos[key])
            self.scene_object_id.append(key)
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
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.speaker_position
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.speaker_position
        return p