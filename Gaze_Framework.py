import json
from typing import Dict, List
import numpy as np
import os, sys
import librosa
from Signal_processing_utils import intensity_from_signal, pitch_from_signal
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_look_at_point_planner import HabituationBasedPlanner, RandomPlanner, PartnerHabituationPlanner
from prototypes.MVP.MVP_eye_head_driver import HeuristicGazeMotionGenerator
from prototypes.EyeCatch.Saccade_model_modified import *
from prototypes.Gaze_aversion_prior.Heuristic_model import *
from prototypes.MVP.MVP_Aversion_saliency_list import *
import pickle
if __name__ == '__main__':

    # inputs
    scene_data_path = "data\look_at_points\simplest_scene.json"

    input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/not_ur_fault"
    # input_folder = "C:/Users/evan1/Documents/neckMovement/data/neck_rotation_values/Merchant_Intro"
    input_file_name = "audio"
    # input_file_name = "audio"
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
    # =============================================================================================
    # =================================== get the saliency maps ===================================
    # =============================================================================================

    # get aversion probability:
    compute_aversion_prob = ComputeAversionProbability(sementic_script, audio)
    aversion_probability_t, aversion_probability_p = compute_aversion_prob.compute()
    # compute aversion saliency map based on aversion probability
    aversion_saliency = AversionSignalDrivenSaliency(scene, audio, sementic_script)
    aversion_saliency.compute_salience(aversion_probability_t, aversion_probability_t)

    # get static saliency maps
    base_saliency = ObjectBasedFixSaliency(scene, audio, sementic_script)
    base_saliency.compute_salience()

    # =============================================================================================
    # ========================== plan scan path based on the saliency maps ========================
    # =============================================================================================

    # get view_target planner
    planner = PartnerHabituationPlanner(base_saliency, audio, sementic_script, scene, 0.8)
    # planner = HabituationBasedPlanner(base_saliency, audio, sementic_script, scene, 0.7)
    # compute the gaze targets and times
    output_times, output_targets = planner.compute()

    #get the output_targets_positions from the scene
    output_target_positions = []
    for i in range(0, len(output_targets)):
        output_target_positions.append(scene.transform_world_to_local(scene.object_pos[output_targets[i]]))

    # get animation curves
    internal_model = InternalModelExact(scene)
    generator = SacccadeGenerator(output_times, output_target_positions, output_targets, internal_model)
    ek, hk, micro_saccade = generator.compute()

    # motion_generator = HeuristicGazeMotionGenerator(scene, sementic_script)
    # ek, hk, micro_saccade = motion_generator.generate_neck_eye_curve(output_times, output_target_positions)
    # out_location = "C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl"
    out_location = "C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/prototype2p2.pkl"

    out = [ek, hk, micro_saccade]
    pickle.dump(out, open(out_location, 'wb'), protocol=2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
