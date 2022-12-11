from prototypes.VirtualClasses import Base_Static_Saliency_List
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from Geometry_Util import rotation_angles_frome_positions
from Signal_processing_utils import interpolate1D
import numpy as np
from scipy.interpolate import interp1d
from Speech_Data_util import Sentence_word_phone_parser
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

class AversionSignalDrivenSaliency(Base_Static_Saliency_List):
    def __init__(self, scene_info: Dietic_Conversation_Gaze_Scene_Info, audio: np.array, script: Sentence_word_phone_parser, sr=44100, dt=0.02):
        self.scene_info: Dietic_Conversation_Gaze_Scene_Info = scene_info
        self._number_of_objects = scene_info.object_pos.shape[0] + scene_info.get_wondering_points().shape[0]
        self._sr = sr
        self._dt = dt # 100 hz
        self._audio_start = 0
        self._audio_end = float(audio.shape[0]) / float(self._sr)
        self._numb_of_frames = int(np.ceil((self._audio_end) / self._dt)) # total number of frames
        # self._audio = audio
        # self._script = script
        self.evaluated = False
        self.map = np.zeros((int(self._numb_of_frames), self._number_of_objects))
        self.map_interp = None
    def compute_salience(self, aversion_prob_time, aversion_prob_val):
        # continue setting salience for all objects
        for j in range(0, self._numb_of_frames):
            for i in range(0, self._number_of_objects):
                if i < self.scene_info.object_pos.shape[0]:
                    self.map[j, i] = 0
                    if self.scene_info.object_type[i] == 5:
                        self.map[j, i] = 1 - interpolate1D(aversion_prob_time, aversion_prob_val, float(j) * self._dt)
                else:
                    if interpolate1D(aversion_prob_time, aversion_prob_val, float(j) * self._dt) < 0.3:
                        self.map[j, i] = 0
                    else:
                        self.map[j, i] = 0.8


    def evaluate_all(self):
        if self.evaluated:
            return self.map
        else:
            self.compute_salience()
            x = np.arange(0, self._numb_of_frames) * self._dt
            self.map_interp = interp1d(x, self.map, axis=0, fill_value="extrapolate")
            self.evaluated = True
            return self.map
        return super(ObjectBasedFixSaliency, self).evaluate_all()
    def evaluate(self, t):
        if self.evaluated:
            return self.map_interp(t)
        else:
            self.compute_salience()
            x = np.arange(0, self._numb_of_frames) * self._dt
            self.map_interp = interp1d(x, self.map, axis=0, fill_value="extrapolate")
            self.evaluated = True
            return self.map_interp(t)
    def get_object_positions(self):
        object_positions = self.scene_info.object_pos
        abstract_positions = self.scene_info.get_wondering_points()
        return np.concatenate([object_positions, abstract_positions], axis=0)

    def plot(self):
        resolution = 5
        out_img = np.zeros((180 * resolution, 360 * resolution))
        out_img[45*resolution:135*resolution, 90*resolution:270*resolution] = 0.1
        rot = rotation_angles_frome_positions(self.scene_info.object_pos)
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Show the image
        ax.imshow(out_img)

        # Now, loop through coord arrays, and create a circle at each x,y pair
        # Show the image
        for object_i in range(len(self.scene_info.scene_object_id)):
            pos_i_global = self.scene_info.object_pos[object_i]
            pos_i_local = self.scene_info.transform_world_to_local((pos_i_global))
            pos_i_local = np.expand_dims(pos_i_local, axis=0)
            rot_i = rotation_angles_frome_positions(pos_i_local)[0]
            print(self.evaluate_all()[0])
            circ = Circle(((int(rot_i[0]) + 180) * resolution, int(rot_i[1] + 90) * resolution), int(40 * self.evaluate_all()[0, object_i]))
            ax.add_patch(circ)

        plt.show()














