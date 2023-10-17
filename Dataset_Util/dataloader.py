import numpy as np
import pickle as pkl
import json
import torch
from torch.utils.data import Dataset
import librosa
import os
import sys
import cv2
import shutil
from matplotlib import pyplot as plt
import csv
from datetime import datetime
from scipy.signal.windows import gaussian
import soundfile as sf
# import utility functions
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Desktop/openpose/python/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/EvansToolBox/Utils/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/Gaze_project/')
# sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox")
# from Geometry_Util import rotation_angles_frome_positions
from Signal_processing_utils import dx_dt

# Dataset for statistical stuff
class ShotDataSet_Selftape111(Dataset):
    def __init__(self, processed_data_path):
        # save dataset root path
        self.data_root_path = processed_data_path

        # load video names
        video_names_path = os.path.join(*[processed_data_path, "metadata.json"])
        self.video_metadata = {}
        with open(video_names_path, mode='r') as f:
            self.video_metadata = json.load(f)["data"]
    def __len__(self):
        return len(self.video_metadata)
    def __getitem__(self, idx):
        file_name = self.video_metadata[idx]["name"]
        fps = self.video_metadata[idx]["fps"]
        output_audio_onscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(0)]) 
        output_audio_offscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(1)]) 
        output_gaze_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"]) 
        output_head_path = os.path.join(*[self.data_root_path, "head", file_name+".pkl"]) 
        output_blinks_path = os.path.join(*[self.data_root_path, "blinks", file_name+".pkl"])
        gaze = pkl.load(open(output_gaze_path, "rb"))
        head = pkl.load(open(output_head_path, "rb"))
        blinks = pkl.load(open(output_blinks_path, "rb"))

        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks]
class SegmentDataset_SelfTape111(Dataset):
    def __init__(self, processed_data_path, videos_included=None, win_length=10, stride_length=5, aversion=False):
        self.aversion = aversion
        # save dataset root path
        self.data_root_path = processed_data_path
        self.count = 0
        
        self.video_metadata = []
        # only consider the videos included
        if not videos_included is None:
            # load video names
            video_names_path = os.path.join(*[self.data_root_path, "metadata.json"])
            temp_video_metadata = {}
            with open(video_names_path, mode='r') as f:
                temp_video_metadata = json.load(f)["data"] 
            for i in temp_video_metadata:
                if i["name"] in videos_included:
                    self.video_metadata.append(i) 
        else:
            # load video names
            video_names_path = os.path.join(*[self.data_root_path, "metadata.json"])
            self.video_metadata = {}
            with open(video_names_path, mode='r') as f:
                self.video_metadata = json.load(f)["data"]  
        # each clip will be 
        clip_metadata = []
        for i in range(0, len(self.video_metadata)):
            metadata = self.video_metadata[i]
            fps = metadata["fps"] # this depends on the video
            sr = metadata["sr"] # they should all be 22500
            video_length = metadata["annotation_length"]
            audio_length = metadata["audio_length"]
            # get the length of the window size, and stride length in frames (fps and sr respectively)
            win_size_audio_per_segment = win_length * sr
            win_size_video_per_segment = int(np.round(win_length * fps))
            stride_length_audio_per_segment = stride_length * sr
            stride_length_video_per_segment = int(np.round(stride_length * fps))
            video_ranges = []
            audio_ranges = []
            # segment the annotation_files
            window_count = np.floor((video_length - (win_size_video_per_segment - stride_length_video_per_segment)) / stride_length_video_per_segment)
            for w in range(0, int(window_count)):
                video_window_start = stride_length_video_per_segment * w
                video_window_end = video_window_start + win_size_video_per_segment
                audio_window_start = stride_length_audio_per_segment * w
                audio_window_end = audio_window_start + win_size_audio_per_segment
                video_ranges.append([video_window_start, video_window_end])
                audio_ranges.append([audio_window_start, audio_window_end])
                self.count = self.count + 1
                clip_metadata.append({"video_range": [video_window_start, video_window_end],
                                      "audio_range": [audio_window_start, audio_window_end],
                                      "fps":fps,
                                      "sr":sr,
                                      "file_name": metadata["name"]})
                # clip_list.append([])
            video_ranges.append([video_length-win_size_video_per_segment, video_length])
            audio_ranges.append([audio_length-win_size_audio_per_segment, audio_length])
            clip_metadata.append({"video_range": video_ranges[-1],
                                  "audio_range": audio_ranges[-1],
                                  "fps":fps,
                                  "sr":sr, 
                                  "file_name": metadata["name"]})
            self.count = self.count + 1
        self.clip_metadata = clip_metadata
        # parse the data into 
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        file_name = self.clip_metadata[idx]["file_name"]
        fps = self.clip_metadata[idx]["fps"]
        v_range = self.clip_metadata[idx]["video_range"]
        a_range = self.clip_metadata[idx]["audio_range"]
        output_audio_onscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(0)]) 
        output_audio_offscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(1)]) 
        output_gaze_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"]) 
        output_head_path = os.path.join(*[self.data_root_path, "head", file_name+".pkl"]) 
        output_blinks_path = os.path.join(*[self.data_root_path, "blinks", file_name+".pkl"])
        output_aversion_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])

        gaze = pkl.load(open(output_gaze_path, "rb"))[v_range[0]:v_range[1]]
        head = pkl.load(open(output_head_path, "rb"))[v_range[0]:v_range[1]]
        blinks = pkl.load(open(output_blinks_path, "rb"))[v_range[0]:v_range[1]]
        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        audio_onscreen = audio_onscreen[a_range[0]:a_range[1]]
        audio_offscreen = audio_offscreen[a_range[0]:a_range[1]]
        if self.aversion:
            aversion = pkl.load(open(output_aversion_path, "rb"))
            return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks, aversion]      
        return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks]  
    
    def total_shot_count(self):
        return len(self.video_metadata)
    
    def get_video(self, idx):
        file_name = self.video_metadata[idx]["name"]
        fps = self.video_metadata[idx]["fps"]
        output_audio_onscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(0)]) 
        output_audio_offscreen_path = os.path.join(*[self.data_root_path, "audio", file_name+"_{}.wav".format(1)]) 
        output_gaze_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"]) 
        output_head_path = os.path.join(*[self.data_root_path, "head", file_name+".pkl"]) 
        output_blinks_path = os.path.join(*[self.data_root_path, "blinks", file_name+".pkl"])
        output_aversion_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])

        gaze = pkl.load(open(output_gaze_path, "rb"))
        head = pkl.load(open(output_head_path, "rb"))
        blinks = pkl.load(open(output_blinks_path, "rb"))
        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        shot_range = self.video_metadata[idx]["video_range"]
        if self.aversion:
            aversion = pkl.load(open(output_aversion_path, "rb"))
            return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks, aversion], [file_name, shot_range]
        return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks], [file_name, shot_range]



# Dataset for deep learning
class Aversion_SelfTap111(Dataset):
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False):
        self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715])
        # save dataset root path
        self.data_root_path = processed_data_path
        # load video names
        video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
        self.metadata = json.load(open(video_names_path, "r"))
        self.all_files_in_set = []
        if videos_included is None:
            videos_included = list(self.metadata.keys())
        for i in videos_included:
            self.all_files_in_set = self.all_files_in_set + self.metadata[i]
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
    def __getitem__(self, idx):
        # idx = self.all_files_in_set[input_index]
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        onscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.word_timing:
            onscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.sentence_and_word_timing:
            onscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
            onscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", "clip_{}.npy".format(idx)])
        # output_target
        output_target = np.load(aversion_label_path)
        # see if we need to concat any thing
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        if self.audio_only:
            missing_frames = output_target.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
            input_vector = np.concatenate([input_audio_on_screen, input_audio_off_screen], axis=1)
            return input_vector, output_target
        input_text_on_screen = np.load(onscreen_text_feature_path)
        input_text_off_screen = np.load(offscreen_text_feature_path)
        if self.sentence_and_word_timing:
            input_text_on_screen1 = np.load(onscreen_text_feature1_path)
            input_text_off_screen1 = np.load(offscreen_text_feature1_path)
            input_text_on_screen2 = np.load(onscreen_text_feature2_path)
            input_text_off_screen2 = np.load(offscreen_text_feature2_path)
            input_text_on_screen = np.concatenate([input_text_on_screen1, input_text_on_screen2], axis = 1)
            input_text_off_screen = np.concatenate([input_text_off_screen1, input_text_off_screen2], axis = 1)
            
        if input_audio_on_screen.shape[0] < input_text_on_screen.shape[0]:
            missing_frames = input_text_on_screen.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
        input_vector_onscreen = np.concatenate([input_audio_on_screen, input_text_on_screen], axis=1)
        input_vector_offscreen = np.concatenate([input_audio_off_screen, input_text_off_screen], axis=1)
        input_vector = np.concatenate([input_vector_onscreen, input_vector_offscreen], axis=1)


        if self.velocity_label:
            vel_output_target = dx_dt(output_target)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            return input_vector, [output_target, vel_output_target]
        return input_vector, output_target
    
# Dataset for deep learning, also with gaze direction
class Aversion_and_Rough_Directions_SelfTap111(Dataset):
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False, simple_dir=False):
        self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715])
        # save dataset root path
        self.data_root_path = processed_data_path
        # load video names
        video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
        self.metadata = json.load(open(video_names_path, "r"))
        self.all_files_in_set = []
        if videos_included is None:
            videos_included = list(self.metadata.keys())
        for i in videos_included:
            self.all_files_in_set = self.all_files_in_set + self.metadata[i]
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        self.simple_dir = simple_dir
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
    def get_audio(self, idx):
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        return input_audio_on_screen, input_audio_off_screen
    def __getitem__(self, idx):
        idx = self.all_files_in_set[idx]
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        onscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.word_timing:
            onscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.sentence_and_word_timing:
            onscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
            onscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", "clip_{}.npy".format(idx)])
        if self.simple_dir:
            aversion_direction_path = os.path.join(*[self.data_root_path, "aversion_direction_simple", "clip_{}.npy".format(idx)])
        else:
            aversion_direction_path = os.path.join(*[self.data_root_path, "aversion_direction", "clip_{}.npy".format(idx)])
        # output_target
        output_target = np.load(aversion_label_path)
        output_target_2 = np.load(aversion_direction_path)
        # see if we need to concat any thing
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        if self.audio_only:
            missing_frames = output_target.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
            input_vector = np.concatenate([input_audio_on_screen, input_audio_off_screen], axis=1)
            return input_vector, [output_target, output_target_2]
        input_text_on_screen = np.load(onscreen_text_feature_path)
        input_text_off_screen = np.load(offscreen_text_feature_path)
        if self.sentence_and_word_timing:
            input_text_on_screen1 = np.load(onscreen_text_feature1_path)
            input_text_off_screen1 = np.load(offscreen_text_feature1_path)
            input_text_on_screen2 = np.load(onscreen_text_feature2_path)
            input_text_off_screen2 = np.load(offscreen_text_feature2_path)
            input_text_on_screen = np.concatenate([input_text_on_screen1, input_text_on_screen2], axis = 1)
            input_text_off_screen = np.concatenate([input_text_off_screen1, input_text_off_screen2], axis = 1)
            
        if input_audio_on_screen.shape[0] < input_text_on_screen.shape[0]:
            missing_frames = input_text_on_screen.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
        input_vector_onscreen = np.concatenate([input_audio_on_screen, input_text_on_screen], axis=1)
        input_vector_offscreen = np.concatenate([input_audio_off_screen, input_text_off_screen], axis=1)
        input_vector = np.concatenate([input_vector_onscreen, input_vector_offscreen], axis=1)


        if self.velocity_label:
            vel_output_target = dx_dt(output_target)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            return input_vector, [output_target, vel_output_target, output_target_2]
        return input_vector, [output_target, output_target_2]

class Aversion_and_Gaze_Directions_SelfTap111(Dataset):
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False):
        self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715])
        # save dataset root path
        self.data_root_path = processed_data_path
        # load video names
        video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
        self.metadata = json.load(open(video_names_path, "r"))
        self.all_files_in_set = []
        if videos_included is None:
            videos_included = list(self.metadata.keys())
        for i in videos_included:
            self.all_files_in_set = self.all_files_in_set + self.metadata[i]
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
    def get_audio(self, idx):
        idx = self.all_files_in_set[idx]
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        return input_audio_on_screen, input_audio_off_screen
    def __getitem__(self, idx):
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        onscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.word_timing:
            onscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.sentence_and_word_timing:
            onscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
            onscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", "clip_{}.npy".format(idx)])
        aversion_direction_path = os.path.join(*[self.data_root_path, "gaze", "clip_{}.npy".format(idx)])
        interlocutor_direction_path = os.path.join(*[self.data_root_path, "interlocutor_direction", "clip_{}.npy".format(idx)])
        # output_target
        output_target = np.load(aversion_label_path)
        gaze_dir = np.load(aversion_direction_path)
        # see if we need to concat any thing
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        if self.audio_only:
            missing_frames = output_target.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
            input_vector = np.concatenate([input_audio_on_screen, input_audio_off_screen], axis=1)
            return input_vector, [output_target, gaze_dir]
        input_text_on_screen = np.load(onscreen_text_feature_path)
        input_text_off_screen = np.load(offscreen_text_feature_path)
        if self.sentence_and_word_timing:
            input_text_on_screen1 = np.load(onscreen_text_feature1_path)
            input_text_off_screen1 = np.load(offscreen_text_feature1_path)
            input_text_on_screen2 = np.load(onscreen_text_feature2_path)
            input_text_off_screen2 = np.load(offscreen_text_feature2_path)
            input_text_on_screen = np.concatenate([input_text_on_screen1, input_text_on_screen2], axis = 1)
            input_text_off_screen = np.concatenate([input_text_off_screen1, input_text_off_screen2], axis = 1)
            
        if input_audio_on_screen.shape[0] < input_text_on_screen.shape[0]:
            missing_frames = input_text_on_screen.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
        input_vector_onscreen = np.concatenate([input_audio_on_screen, input_text_on_screen], axis=1)
        input_vector_offscreen = np.concatenate([input_audio_off_screen, input_text_off_screen], axis=1)
        input_vector = np.concatenate([input_vector_onscreen, input_vector_offscreen], axis=1)
        interlocutor_direction = np.load(interlocutor_direction_path)

        if self.velocity_label:
            vel_output_target = dx_dt(output_target)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            return input_vector, [output_target, vel_output_target, gaze_dir]
        return input_vector, [output_target, gaze_dir, interlocutor_direction]
