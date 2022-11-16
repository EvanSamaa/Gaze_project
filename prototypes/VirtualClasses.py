import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from Signal_processing_utils import *
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
import os

class Base_Static_Saliency_List(ABC):
    """
    abstract method for defining saliency map, which wiill be used to determine probability of look at points
    over time
    """

    @abstractmethod
    def __init__(self, scene_info, audio: np.array, script: Sentence_word_phone_parser, sr=44100):
        pass

    @abstractmethod
    def compute_salience(self):
        """
        compute saliency map. update self.map
        :return: Nothing
        """
        pass

    @abstractmethod
    def evaluate(self, time):
        """
        Evaluate the method at the desired time
        :param time:
        :return:
        """
        pass

    @abstractmethod
    def evaluate_all(self):
        """
        obtain a 2D array of shape (time, number of scene objects)
        :return:
        """
        pass
class Base_Dynamic_Saliency_List(ABC):
    """
    abstract method for defining saliency map, which wiill be used to determine probability of look at points
    over time
    """

    @property
    def map(self):
        pass
    @property
    def evaluated(self):
        pass

    @abstractmethod
    def __init__(self, scene_info, audio: np.array, script: Sentence_word_phone_parser, sr=44100):
        pass

    @abstractmethod
    def compute_salience(self):
        """
        compute saliency map. update self.map
        :return: Nothing
        """
        pass

    def evaluate_all(self, t_start, t_end):
        """
        obtain a 2D array of shape (time, number of scene objects)
        :param t_start:
        :param t_end:
        :return: the
        """
        if self.evaluated:
            return self.map
        else:
            self.compute()
            return self.map






if __name__ == "__main__":
    scene_data_path = "data\look_at_points\simplest_scene.json"

    input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/CNN"
    input_file_name = "cnn_borderOneGuy"

    # get scene data
    scene_data_path = "../data/look_at_points/simplest_scene.json"
    scene = Gaze_Scene_Info(scene_data_path)

    # get audio+script+alignment data
    audio_location = os.path.join(input_folder, input_file_name + ".wav")
    script_location = os.path.join(input_folder, input_file_name + ".txt")
    praatscript_location = os.path.join(input_folder, input_file_name + "_PraatOutput.txt")

    audio, sr = librosa.load(audio_location, sr=44100)
    intensity = intensity_from_signal(audio)
    pitch = pitch_from_signal(audio)

    # get alignment related things
    sementic_structure = Sentence_word_phone_parser(praatscript_location, script_location)
    tim = Base_Aversion_prediction(audio, sementic_structure)
