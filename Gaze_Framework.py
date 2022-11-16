import json
from typing import Dict, List
import numpy as np
import os, sys
import librosa
from Signal_processing_utils import intensity_from_signal, pitch_from_signal
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.InputDataStructures import Gaze_Scene_Info
from prototypes.MVP.MVP_aversion_saliency_map import ObjectBasedFixSaliency
if __name__ == '__main__':

    # inputs
    scene_data_path = "data\look_at_points\simplest_scene.json"

    input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/CNN"
    input_file_name = "cnn_borderOneGuy"

    # get scene data
    scene_data_path = "data\look_at_points\simplest_scene.json"
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

    # get time_varying saliency maps
    base_saliency = ObjectBasedFixSaliency(scene, audio, sementic_structure)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
