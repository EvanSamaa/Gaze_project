import json
from typing import Dict, List
import numpy as np
import os, sys
import librosa
from Signal_processing_utils import intensity_from_signal, pitch_from_signal
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_look_at_point_planner import HabituationBasedPlanner
if __name__ == '__main__':

    # inputs
    scene_data_path = "data\look_at_points\simplest_scene.json"

    input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/CNN"
    input_folder = "C:/Users/evan1/Documents/neckMovement/data/neck_rotation_values/news_anchor_1"
    input_file_name = "cnn_borderOneGuy"
    input_file_name = "audio"
    # get scene data
    scene_data_path = "data\look_at_points\simplest_scene.json"
    scene = Dietic_Conversation_Gaze_Scene_Info(scene_data_path)

    # get audio+script+alignment data
    audio_location = os.path.join(input_folder, input_file_name + ".wav")
    script_location = os.path.join(input_folder, input_file_name + ".txt")
    praatscript_location = os.path.join(input_folder, input_file_name + "_PraatOutput.txt")

    audio, sr = librosa.load(audio_location, sr=44100)
    # intensity = intensity_from_signal(audio)
    # pitch = pitch_from_signal(audio)

    # get alignment related things
    sementic_script = Sentence_word_phone_parser(praatscript_location, script_location)

    # get static saliency maps
    base_saliency = ObjectBasedFixSaliency(scene, audio, sementic_script)
    base_saliency.compute_salience()
    # get planner
    planner = HabituationBasedPlanner(base_saliency, audio, sementic_script, scene)
    planner.compute()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
