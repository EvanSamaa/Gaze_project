from prototypes.VirtualClasses import Base_Saliency_List
from prototypes.InputDataStructures import Gaze_Scene_Info
import numpy as np
from Speech_Data_util import Sentence_word_phone_parser

class ObjectBasedFixSaliency(Base_Saliency_List):
    def __init__(self, scene_info: Gaze_Scene_Info, audio: np.array, script: Sentence_word_phone_parser, sr=44100):
        self.scene_info = scene_info
        self._number_of_objects = len(scene_info.object_pos)
        self._sr = 44100
        self._dt = 0.02 # 100 hz
        self._audio_start = 0
        self._audio_end = float(audio.shape[0]) / float(self._sr)
        # self._numb_of_frames = np.ceil((self._audio_end) / self._dt) # total number of frames
        # self._audio = audio
        # self._script = script
        self.evaluated = False
        self.map = np.zeros((self._numb_of_frames, self._number_of_objects))
    def compute_salience(self, attended_to_id):
        # set the salience of the conversation partner to 1 by default
        for i in range(0, len(self.scene_info.object_type)):
            if self.scene_info.object_type == 1:
                self.map[0, i] = self.scene_info.object_interest * 10;
            else:
                self.map[0, i] = self.scene_info.object_interest;
        # continue setting salience for all objects
        for i in range(1, self._number_of_frames):
            self.map[i] = self.map[0]







