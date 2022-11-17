from prototypes.VirtualClasses import Base_Static_Saliency_List
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
import numpy as np
from scipy.interpolate import interp1d
from Speech_Data_util import Sentence_word_phone_parser

class ObjectBasedFixSaliency(Base_Static_Saliency_List):
    def __init__(self, scene_info: Dietic_Conversation_Gaze_Scene_Info, audio: np.array, script: Sentence_word_phone_parser, sr=44100):
        self.scene_info: Dietic_Conversation_Gaze_Scene_Info = scene_info
        self._number_of_objects = len(scene_info.object_pos)
        self._sr = 44100
        self._dt = 0.02 # 100 hz
        self._audio_start = 0
        self._audio_end = float(audio.shape[0]) / float(self._sr)
        self._numb_of_frames = int(np.ceil((self._audio_end) / self._dt)) # total number of frames
        # self._audio = audio
        # self._script = script
        self.evaluated = False
        self.map = np.zeros((int(self._numb_of_frames), self._number_of_objects))
        self.map_interp = None
    def compute_salience(self):
        print(self.scene_info.object_interest)
        # set the salience of the conversation partner to 1 by default
        for i in range(0, len(self.scene_info.object_type)):
            obj_id = self.scene_info.scene_object_id[i]
            if self.scene_info.object_type == 1:
                self.map[0, i] = self.scene_info.object_interest[obj_id]
            else:
                self.map[0, i] = self.scene_info.object_interest[obj_id] * self.scene_info.object_distance_to_listener[obj_id]
        # continue setting salience for all objects
        for i in range(1, self._numb_of_frames):
            self.map[i] = self.map[0]
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











