import json
from typing import Dict, List
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import librosa
from Signal_processing_utils import intensity_from_signal, pitch_from_signal, sparse_key_smoothing
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from prototypes.MVP.MVP_Aversion_saliency_list import AversionSignalDrivenSaliency, CTSAversionSignalDrivenSaliency
from prototypes.MVP.MVP_look_at_point_planner import HabituationBasedPlanner, RandomPlanner, PartnerHabituationPlanner
from prototypes.MVP.MVP_eye_head_driver import HeuristicGazeMotionGenerator
from prototypes.EyeCatch.Saccade_model_modified import SacccadeGenerator, InternalModelExact
from prototypes.Gaze_aversion_prior.Heuristic_model import *
from prototypes.Boccignone2020.Gaze_target_planner import Scavenger_based_planner
from prototypes.Boccignone2020.Improved_gaze_target_planner import Scavenger_planner_with_nest
from prototypes.JaliNeck.JaliNeck import NeckCurve
from prototypes.Gaze_aversion_prior.Ribhav_model import predict_aversion
import pickle
import math
if __name__ == '__main__':
    np.random.seed(0)
    # inputs
    scene_data_path = "data/look_at_points/simplest_scene.json"
    # input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/not_ur_fault"
    input_folder = "/Volumes/EVAN_DISK/MASC/JALI_neck/data/neck_rotation_values/not_ur_fault"
    # input_folder = "C:/Users/evan1/Documents/neckMovement/data/neck_rotation_values/Sarah"

    # input_file_name = "audio"
    input_file_name = "audio"
    # get scene data
    scene_data_path = "data/look_at_points/simplest_scene.json"
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
    # based on voiced pauses
    aversion_probability_t, aversion_probability_p = compute_aversion_prob.compute()
    # based on deep learning
    dp_aversion_probability_t, dp_aversion_probability_p = predict_aversion(audio_location, dt=0.02)
    # compute aversion saliency map based on aversion probability
    aversion_saliency = AversionSignalDrivenSaliency(scene, audio, sementic_script, dt=0.02)
    aversion_saliency.compute_salience(aversion_probability_t, aversion_probability_p)
    # get static saliency maps
    base_saliency = ObjectBasedFixSaliency(scene, audio, sementic_script)
    base_saliency.compute_salience()
    # get the aversion saliency based on DL prediction

    aversion_saliency_audio = CTSAversionSignalDrivenSaliency(scene, dp_aversion_probability_t[:base_saliency.map.shape[0]], dp_aversion_probability_p[:base_saliency.map.shape[0]])
    aversion_saliency_audio.compute_salience()
    # =============================================================================================
    # ========================== plan scan path based on the saliency maps ========================
    # =============================================================================================
    planner = Scavenger_planner_with_nest([base_saliency, aversion_saliency_audio], scene)
    # planner = Scavenger_planner_with_nest([base_saliency, aversion_saliency], scene)
    output_times, output_targets = planner.compute(scene.object_type.argmax())
    # get view_target planner
    # planner = PartnerHabituationPlanner(base_saliency, audio, sementic_script, scene, 0.8)
    # planner = HabituationBasedPlanner(base_saliency, audio, sementic_script, scene, 0.7)
    # compute the gaze targets and times
    # output_times, output_targets = planner.compute()
    #get the output_targets_positions from the scene
    output_target_positions = []
    wondering_positions = scene.get_wondering_points()
    for i in range(0, len(output_targets)):
        if output_targets[i] < scene.object_pos.shape[0]:
            # the real scene objects have physical location in world coor dinate space
            output_target_positions.append(scene.transform_world_to_local(scene.object_pos[output_targets[i]]))
        else:
            # the virtual look-at directions (i.e. pondering locations)are in local coordinate space
            output_target_positions.append(aversion_saliency.get_object_positions()[output_targets[i]])
    # get animation curves
    internal_model = InternalModelExact(scene)
    generator = SacccadeGenerator(output_times, output_target_positions, output_targets, internal_model)
    ek, hk, micro_saccade = generator.compute()
    conversational_neck = NeckCurve(audio_location)
    jali_neck_output = conversational_neck.compute_curve()

    # arr = np.array(ek)
    # arr = arr[0, :, 1:]
    # partner = scene.transform_world_to_local(scene.object_pos[scene.get_conversation_partner_id()[0]])
    # partner = np.expand_dims(partner, axis=0)
    # distance_with_goal = arr - partner
    # plt.plot(distance_with_goal)
    # plt.show()
    blend_weight = []
    for i in range(1, len(hk[0])-1):
        velocity = math.sqrt((hk[0][i][1]-hk[0][i-1][1])**2 + (hk[0][i-1][2]-hk[0][i][2])**2)
        blend_weight.append([hk[0][i][0], 1 - min(1, velocity/0.75)])
    # motion_generator = HeuristicGazeMotionGenerator(scene, sementic_script)
    # ek, hk, micro_saccade = motion_generator.generate_neck_eye_curve(output_times, output_target_positions)
    # out_location = "C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl"
    # out_location = "C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/prototype2p2.pkl"
    out_location = "data/prototype2p2.pkl"
    out = [ek, hk, micro_saccade, jali_neck_output, []]
    pickle.dump(out, open(out_location, 'wb'), protocol=2)
    print("done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
