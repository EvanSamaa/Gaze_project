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
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False, mel_only=False):
        self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715])
        if mel_only:
            self.filler = self.filler[0:13]
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
        self.all_files_in_set = list(set(self.all_files_in_set))
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        self.mel_only = mel_only
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
    def __getitem__(self, input_index):
        idx = self.all_files_in_set[input_index]
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
        if self.mel_only:
            input_audio_on_screen = input_audio_on_screen[:, 0:13]
            input_audio_off_screen = input_audio_off_screen[:, 0:13]
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
# Dataset for deep learning
class Aversion_SelfTap111_original(Dataset):
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False, mel_only=False):
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
        self.all_files_in_set = list(set(self.all_files_in_set))
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        self.mel_only = mel_only
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
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
        # output_target
        output_target = np.load(aversion_label_path)
        # see if we need to concat any thing
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        if self.mel_only:
            input_audio_on_screen = input_audio_on_screen[0:13]
            input_audio_off_screen = input_audio_off_screen[0:13]
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
        # idx = self.all_files_in_set[idx]
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

class Aversion_and_Gaze_Directions_SelfTap111_original(Dataset):
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
        # idx = self.all_files_in_set[idx]
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        return input_audio_on_screen, input_audio_off_screen
    def __getitem__(self, idx):
        # idx = self.all_files_in_set[idx]
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

import random
class Runtime_parsing_Aversion_SelfTape111(Dataset):
    def __init__(self, processed_data_path, videos_included=None, prev_dataset=None, pos_labels=True, long_aversion_only=False, shuffle=True, window_length=250, with_gaze=False, normalize_MFCC=False, apply_frequency_mask=False, apply_time_mask=False):
        torch.set_default_tensor_type(torch.DoubleTensor)
        if prev_dataset is None:        
            self.data_root_path = processed_data_path
            self.shuffle = shuffle
            self.pos_labels = pos_labels
            self.window_length = window_length
            self.with_gaze = with_gaze
            self.long_aversion_only = long_aversion_only
            video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
            self.metadata = json.load(open(video_names_path, "r"))
            self.all_files_in_set = []
            if videos_included is None:
                videos_included = list(self.metadata.keys())
            self.all_files_in_set = videos_included
            self.gaussian_window = gaussian(5, 1)
            self.normalize_MFCC = normalize_MFCC
            self.apply_frequency_mask = apply_frequency_mask
            self.apply_time_mask = apply_time_mask
            # load all input features and aversionl labels to memory
            self.input_features = []
            self.aversion_labels = []
            self.velocity_labels = []
            self.gaze_labels = []
            self.interlocutor_positions = []
            self.load_IO_features_to_memory()
            # generate a map to map the index of the dataset to the video
            self.map = {}
            self.dataset_length = 0
            self.parse_dataset()
            # generate filler for input features:
            self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0])
            self.filler_back = np.concatenate([self.filler, np.zeros(6), self.filler, np.zeros(6)])
            if self.pos_labels:
                self.filler_back = np.concatenate([self.filler, np.zeros(20), self.filler, np.zeros(20)])
        else:
            self.data_root_path = prev_dataset.data_root_path
            self.shuffle = prev_dataset.shuffle
            self.pos_labels = prev_dataset.pos_labels
            self.window_length = prev_dataset.window_length
            self.window_length = window_length
            self.long_aversion_only = prev_dataset.long_aversion_only
            self.all_files_in_set = prev_dataset.all_files_in_set
            self.gaussian_window = prev_dataset.gaussian_window
            self.input_features = prev_dataset.input_features
            self.aversion_labels = prev_dataset.aversion_labels
            self.velocity_labels = prev_dataset.velocity_labels
            self.with_gaze = prev_dataset.with_gaze
            self.map = prev_dataset.map
            self.dataset_length = prev_dataset.dataset_length
            self.filler = prev_dataset.filler
            self.filler_back = prev_dataset.filler_back
            self.normalize_MFCC = prev_dataset.normalize_MFCC
            self.apply_time_mask = prev_dataset.apply_time_mask
            self.apply_frequency_mask = prev_dataset.apply_frequency_mask
            self.parse_dataset()
    def __len__(self):
        return self.dataset_length
    def parse_dataset(self):
        self.map = {}
        self.dataset_length = 0
        counter = 0
        for i in range(len(self.input_features)):
            # for randomly cutting the video
            random_offset = np.random.randint(0, self.window_length/2)
            # code starts here

            video_length = self.input_features[i].shape[0] - random_offset # if we start going through the video from the random offset, we will have this many frames left
            stride_length_video_per_segment = int(np.round(self.window_length/2))
            window_count = np.floor((video_length - (self.window_length - stride_length_video_per_segment)) / stride_length_video_per_segment)
            if self.input_features[i].shape[0] <= 25:
                continue
            if video_length <= 0:
                continue
            # add all the windows except the last window
            for w in range(0, int(window_count)):
                # start will be some offset away from the start
                video_window_start = stride_length_video_per_segment * w + random_offset
                video_window_end = video_window_start + self.window_length
                window_range = [video_window_start, video_window_end]
                self.map[counter] = [i, window_range]
                counter = counter + 1
            self.map[counter] = [i, [max(0, video_length-self.window_length), video_length]]
            counter += 1
        self.dataset_length = counter
    def time_mask(self, spec, T=30, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[0]
        for i in range(0, num_masks):
            # I only have 250 ish samples so I'm masking 20 max
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[t_zero:mask_end] = 0
            else: cloned[t_zero:mask_end] = cloned.mean()
        return cloned
    def freq_mask(self, spec, F=5, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
            else: cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned
    def load_IO_features_to_memory(self):
        for file_name in self.all_files_in_set:
            # get the aversion labels from the disk
            if self.long_aversion_only:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "long_aversion_label", file_name+".pkl"])
            else:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])
            if self.with_gaze:
                gaze_label_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"])
                self.gaze_labels.append(pkl.load(open(gaze_label_path, "rb")))
                interlocutor_position_path = os.path.join(*["/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset/", "tinterlocutor_direction", file_name+".pkl"])
                self.interlocutor_positions.append(pkl.load(open(interlocutor_position_path, "rb")))
            output_aversion_label = pkl.load(open(output_aversion_label_path, "rb"))

            # get the input features from the disk 
            on_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_0.pkl"]) 
            off_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_1.pkl"])
            on_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_0.pkl"])
            off_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_1.pkl"])
            on_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_0.pkl"])
            off_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_1.pkl"])
            
            # load the input features from the disk
            on_screen_sentence_timing = pkl.load(open(on_screen_sentence_timing_path, "rb"))
            off_screen_sentence_timing = pkl.load(open(off_screen_sentence_timing_path, "rb"))
            on_screen_mfcc = pkl.load(open(on_screen_mfcc_path, "rb"))
            off_screen_mfcc = pkl.load(open(off_screen_mfcc_path, "rb"))
            if self.normalize_MFCC:
                mean = np.mean(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.std(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.where(std <= 1E-8, 1, std)
                on_screen_mfcc = (on_screen_mfcc - mean) / std  
                off_screen_mfcc = (off_screen_mfcc - mean) / std
                # now this is normalized to 0 mean and 1 std
            if self.pos_labels:
                on_screen_pos = pkl.load(open(on_screen_pos_path, "rb"))
                off_screen_pos = pkl.load(open(off_screen_pos_path, "rb")) 
            if on_screen_mfcc.shape[0] <= 50:
                continue
            # get input features
            input_features_on_screen = np.concatenate([on_screen_mfcc, on_screen_sentence_timing], axis=1)
            input_features_off_screen = np.concatenate([off_screen_mfcc, off_screen_sentence_timing], axis=1)
            if self.pos_labels: # the last 14 features are the POS tags
                input_features_on_screen = np.concatenate([input_features_on_screen, on_screen_pos], axis=1)
                input_features_off_screen = np.concatenate([input_features_off_screen, off_screen_pos], axis=1)
            input_feature = np.concatenate([input_features_on_screen, input_features_off_screen], axis=1)
            vel_output_target = dx_dt(output_aversion_label)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            self.input_features.append(input_feature)
            self.aversion_labels.append(output_aversion_label)
            self.velocity_labels.append(vel_output_target)
    def __getitem__(self, idx):
        # pad all audio to 250 frames
        input_feature = self.input_features[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        aversion_label = self.aversion_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        velocity_label = self.velocity_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        # print(self.window_length, self.map[idx][1][0], self.map[idx][1][1], self.input_features[self.map[idx][0]].shape, self.aversion_labels[self.map[idx][0]].shape, self.velocity_labels[self.map[idx][0]].shape)
        if input_feature.shape[0] < self.window_length:
            missing_frames = self.window_length - input_feature.shape[0]
            padding = np.tile(np.expand_dims(self.filler_back, axis=0), [missing_frames, 1])
            input_feature = np.concatenate([input_feature, padding], axis=0)
            final_aversion_frame = aversion_label[-1]
            repeated_final_aversion_frame = np.tile(np.expand_dims(final_aversion_frame, axis=0), [missing_frames])
            aversion_label = np.concatenate([aversion_label, repeated_final_aversion_frame], axis=0)
            velocity_label = np.concatenate([velocity_label, np.zeros(missing_frames)], axis=0)  

        
        input_feature = torch.from_numpy(input_feature).double()
        if self.apply_time_mask and self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(self.time_mask(input_feature[:, 0:26]))
            input_feature[:, 46:72] = self.freq_mask(self.time_mask(input_feature[:, 46:72]))
        elif self.apply_time_mask:
            input_feature[:, 0:26] = self.time_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.time_mask(input_feature[:, 46:72])
        elif self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.freq_mask(input_feature[:, 46:72])
        aversion_label = torch.from_numpy(aversion_label).double()
        velocity_label = torch.from_numpy(velocity_label).double()
        if self.with_gaze:
            gaze = self.gaze_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
            return input_feature, [aversion_label, gaze, self.interlocutor_positions[self.map[idx][0]], velocity_label] 
        return input_feature, [aversion_label, velocity_label]      
        
class Runtime_parsing_Aversion_SelfTape111_within_subject(Dataset):
    def __init__(self, processed_data_path, proportion_included=[0, 0.9], prev_dataset=None, pos_labels=True, long_aversion_only=False, shuffle=True, window_length=250, with_gaze=False):
        if prev_dataset is None:        
            self.data_root_path = processed_data_path
            self.shuffle = shuffle
            self.pos_labels = pos_labels
            self.window_length = window_length
            self.with_gaze = with_gaze
            self.long_aversion_only = long_aversion_only
            video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
            self.metadata = json.load(open(video_names_path, "r"))
            self.all_files_in_set = list(self.metadata.keys())
            self.proportion_included = proportion_included
            self.gaussian_window = gaussian(5, 1)
            # load all input features and aversionl labels to memory
            self.input_features = []
            self.aversion_labels = []
            self.velocity_labels = []
            self.gaze_labels = []
            self.interlocutor_positions = []
            self.load_IO_features_to_memory()
            # generate a map to map the index of the dataset to the video
            self.map = {}
            self.dataset_length = 0
            self.parse_dataset()
            # generate filler for input features:
            self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0])
            self.filler_back = np.concatenate([self.filler, np.zeros(6), self.filler, np.zeros(6)])
            if self.pos_labels:
                self.filler_back = np.concatenate([self.filler, np.zeros(20), self.filler, np.zeros(20)])
        else:
            self.data_root_path = prev_dataset.data_root_path
            self.shuffle = prev_dataset.shuffle
            self.pos_labels = prev_dataset.pos_labels
            self.window_length = prev_dataset.window_length
            self.window_length = window_length
            self.long_aversion_only = prev_dataset.long_aversion_only
            self.proportion_included = prev_dataset.proportion_included
            self.all_files_in_set = prev_dataset.all_files_in_set
            self.gaussian_window = prev_dataset.gaussian_window
            self.input_features = prev_dataset.input_features
            self.aversion_labels = prev_dataset.aversion_labels
            self.velocity_labels = prev_dataset.velocity_labels
            self.with_gaze = prev_dataset.with_gaze
            self.map = prev_dataset.map
            self.dataset_length = prev_dataset.dataset_length
            self.filler = prev_dataset.filler
            self.filler_back = prev_dataset.filler_back
            self.parse_dataset()
    def __len__(self):
        return self.dataset_length
    def parse_dataset(self):
        self.map = {}
        self.dataset_length = 0
        counter = 0
        for i in range(len(self.input_features)):
            # for randomly cutting the video
            random_offset = np.random.randint(0, self.window_length/2)
            # code starts here

            video_length = self.input_features[i].shape[0] - random_offset # if we start going through the video from the random offset, we will have this many frames left
            stride_length_video_per_segment = int(np.round(self.window_length/2))
            window_count = np.floor((video_length - (self.window_length - stride_length_video_per_segment)) / stride_length_video_per_segment)
            if self.input_features[i].shape[0] <= 25:
                continue
            if video_length <= 0:
                continue
            # add all the windows except the last window
            for w in range(0, int(window_count)):
                if w  / window_count > self.proportion_included[0] and w / window_count < self.proportion_included[1]:
                    # start will be some offset away from the start
                    video_window_start = stride_length_video_per_segment * w + random_offset
                    video_window_end = video_window_start + self.window_length
                    window_range = [video_window_start, video_window_end]
                    self.map[counter] = [i, window_range]
                    counter = counter + 1
        self.dataset_length = counter
    def load_IO_features_to_memory(self):
        for file_name in self.all_files_in_set:

            # get the aversion labels from the disk
            if self.long_aversion_only:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "long_aversion_label", file_name+".pkl"])
            else:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])
            if self.with_gaze:
                gaze_label_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"])
                self.gaze_labels.append(pkl.load(open(gaze_label_path, "rb")))
                interlocutor_position_path = os.path.join(*["/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset/", "tinterlocutor_direction", file_name+".pkl"])
                self.interlocutor_positions.append(pkl.load(open(interlocutor_position_path, "rb")))
            output_aversion_label = pkl.load(open(output_aversion_label_path, "rb"))

            # get the input features from the disk 
            on_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_0.pkl"]) 
            off_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_1.pkl"])
            on_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_0.pkl"])
            off_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_1.pkl"])
            on_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_0.pkl"])
            off_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_1.pkl"])
            
            # load the input features from the disk
            on_screen_sentence_timing = pkl.load(open(on_screen_sentence_timing_path, "rb"))
            off_screen_sentence_timing = pkl.load(open(off_screen_sentence_timing_path, "rb"))
            on_screen_mfcc = pkl.load(open(on_screen_mfcc_path, "rb"))
            off_screen_mfcc = pkl.load(open(off_screen_mfcc_path, "rb"))
            if self.pos_labels:
                on_screen_pos = pkl.load(open(on_screen_pos_path, "rb"))
                off_screen_pos = pkl.load(open(off_screen_pos_path, "rb")) 
            if on_screen_mfcc.shape[0] <= 50:
                continue
            # get input features
            input_features_on_screen = np.concatenate([on_screen_mfcc, on_screen_sentence_timing], axis=1)
            input_features_off_screen = np.concatenate([off_screen_mfcc, off_screen_sentence_timing], axis=1)
            if self.pos_labels: # the last 14 features are the POS tags
                input_features_on_screen = np.concatenate([input_features_on_screen, on_screen_pos], axis=1)
                input_features_off_screen = np.concatenate([input_features_off_screen, off_screen_pos], axis=1)
            input_feature = np.concatenate([input_features_on_screen, input_features_off_screen], axis=1)
            vel_output_target = dx_dt(output_aversion_label)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            self.input_features.append(input_feature)
            self.aversion_labels.append(output_aversion_label)
            self.velocity_labels.append(vel_output_target)
    def __getitem__(self, idx):
        # pad all audio to 250 frames
        input_feature = self.input_features[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        aversion_label = self.aversion_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        velocity_label = self.velocity_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        # print(self.window_length, self.map[idx][1][0], self.map[idx][1][1], self.input_features[self.map[idx][0]].shape, self.aversion_labels[self.map[idx][0]].shape, self.velocity_labels[self.map[idx][0]].shape)

        if input_feature.shape[0] < self.window_length:
            missing_frames = self.window_length - input_feature.shape[0]
            padding = np.tile(np.expand_dims(self.filler_back, axis=0), [missing_frames, 1])
            input_feature = np.concatenate([input_feature, padding], axis=0)
            final_aversion_frame = aversion_label[-1]
            repeated_final_aversion_frame = np.tile(np.expand_dims(final_aversion_frame, axis=0), [missing_frames])
            aversion_label = np.concatenate([aversion_label, repeated_final_aversion_frame], axis=0)
            velocity_label = np.concatenate([velocity_label, np.zeros(missing_frames)], axis=0)  
        input_feature = torch.from_numpy(input_feature).double()
        aversion_label = torch.from_numpy(aversion_label).double()
        velocity_label = torch.from_numpy(velocity_label).double()
        if self.with_gaze:
            gaze = self.gaze_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
            return input_feature, [aversion_label, gaze, self.interlocutor_positions[self.map[idx][0]], velocity_label] 
        return input_feature, [aversion_label, velocity_label]      

import random
class Runtime_parsing_Aversion_SelfTape111_validation_leak(Dataset):
    def __init__(self, processed_data_path, videos_included=None, prev_dataset=None, pos_labels=True, long_aversion_only=False, shuffle=True, window_length=250, with_gaze=False, normalize_MFCC=False, apply_frequency_mask=False, apply_time_mask=False, percent_leaked=0.10):
        torch.set_default_tensor_type(torch.DoubleTensor)
        if prev_dataset is None:        
            self.data_root_path = processed_data_path
            self.shuffle = shuffle
            self.pos_labels = pos_labels
            self.percent_leaked = percent_leaked
            self.window_length = window_length
            self.with_gaze = with_gaze
            self.long_aversion_only = long_aversion_only
            video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
            self.metadata = json.load(open(video_names_path, "r"))
            self.all_files_in_set = []
            if videos_included is None:
                videos_included = list(self.metadata.keys())
            self.all_files_val_and_trian = list(self.metadata.keys())
            self.train_set = []
            self.all_files_in_set = videos_included
            self.gaussian_window = gaussian(5, 1)
            self.normalize_MFCC = normalize_MFCC
            self.apply_frequency_mask = apply_frequency_mask
            self.apply_time_mask = apply_time_mask
            # load all input features and aversionl labels to memory
            self.input_features = []
            self.aversion_labels = []
            self.velocity_labels = []
            self.gaze_labels = []
            self.interlocutor_positions = []
            self.load_IO_features_to_memory()
            # generate a map to map the index of the dataset to the video
            self.map = {}
            self.dataset_length = 0
            self.parse_dataset()
            # generate filler for input features:
            self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0])
            self.filler_back = np.concatenate([self.filler, np.zeros(6), self.filler, np.zeros(6)])
            if self.pos_labels:
                self.filler_back = np.concatenate([self.filler, np.zeros(20), self.filler, np.zeros(20)])
        else:
            self.data_root_path = prev_dataset.data_root_path
            self.shuffle = prev_dataset.shuffle
            self.pos_labels = prev_dataset.pos_labels
            self.window_length = prev_dataset.window_length
            self.window_length = window_length
            self.long_aversion_only = prev_dataset.long_aversion_only
            self.all_files_in_set = prev_dataset.all_files_in_set
            self.gaussian_window = prev_dataset.gaussian_window
            self.input_features = prev_dataset.input_features
            self.aversion_labels = prev_dataset.aversion_labels
            self.velocity_labels = prev_dataset.velocity_labels
            self.with_gaze = prev_dataset.with_gaze
            self.map = prev_dataset.map
            self.dataset_length = prev_dataset.dataset_length
            self.filler = prev_dataset.filler
            self.filler_back = prev_dataset.filler_back
            self.normalize_MFCC = prev_dataset.normalize_MFCC
            self.apply_time_mask = prev_dataset.apply_time_mask
            self.apply_frequency_mask = prev_dataset.apply_frequency_mask
            self.all_files_val_and_trian = prev_dataset.all_files_val_and_trian
            self.train_set = prev_dataset.train_set
            self.percent_leaked = prev_dataset.percent_leaked
            self.parse_dataset()
    def __len__(self):
        return self.dataset_length
    def parse_dataset(self):
        self.map = {}
        self.dataset_length = 0
        counter = 0
        for i in range(len(self.input_features)):
            # for randomly cutting the video
            random_offset = np.random.randint(0, self.window_length/2)
            # code starts here
            video_length = self.input_features[i].shape[0] - random_offset # if we start going through the video from the random offset, we will have this many frames left
            stride_length_video_per_segment = int(np.round(self.window_length/2))
            window_count = np.floor((video_length - (self.window_length - stride_length_video_per_segment)) / stride_length_video_per_segment)
            if self.input_features[i].shape[0] <= 25:
                continue
            if video_length <= 0:
                continue
            # add all the windows except the last window
            if i in self.train_set:
                for w in range(0, int(window_count)):
                    # start will be some offset away from the start
                    video_window_start = stride_length_video_per_segment * w + random_offset
                    video_window_end = video_window_start + self.window_length
                    window_range = [video_window_start, video_window_end]
                    self.map[counter] = [i, window_range]
                    counter = counter + 1
                self.map[counter] = [i, [max(0, video_length-self.window_length), video_length]]
                counter += 1
            else:
                for w in range(0, int(window_count)):
                    if w/float(window_count) <= self.percent_leaked:
                        # start will be some offset away from the start
                        video_window_start = stride_length_video_per_segment * w + random_offset
                        video_window_end = video_window_start + self.window_length
                        window_range = [video_window_start, video_window_end]
                        self.map[counter] = [i, window_range]
                        counter = counter + 1
        self.dataset_length = counter
    def time_mask(self, spec, T=30, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[0]
        for i in range(0, num_masks):
            # I only have 250 ish samples so I'm masking 20 max
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[t_zero:mask_end] = 0
            else: cloned[t_zero:mask_end] = cloned.mean()
        return cloned
    def freq_mask(self, spec, F=5, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
            else: cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned
    def load_IO_features_to_memory(self):
        counter = 0
        for file_name in self.all_files_val_and_trian:
            if file_name in self.all_files_in_set:
                self.train_set.append(counter)
            # get the aversion labels from the disk
            if self.long_aversion_only:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "long_aversion_label", file_name+".pkl"])
            else:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])
            if self.with_gaze:
                gaze_label_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"])
                self.gaze_labels.append(pkl.load(open(gaze_label_path, "rb")))
                interlocutor_position_path = os.path.join(*["/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset/", "tinterlocutor_direction", file_name+".pkl"])
                self.interlocutor_positions.append(pkl.load(open(interlocutor_position_path, "rb")))
            output_aversion_label = pkl.load(open(output_aversion_label_path, "rb"))

            # get the input features from the disk 
            on_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_0.pkl"]) 
            off_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_1.pkl"])
            on_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_0.pkl"])
            off_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_1.pkl"])
            on_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_0.pkl"])
            off_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_1.pkl"])
            
            # load the input features from the disk
            on_screen_sentence_timing = pkl.load(open(on_screen_sentence_timing_path, "rb"))
            off_screen_sentence_timing = pkl.load(open(off_screen_sentence_timing_path, "rb"))
            on_screen_mfcc = pkl.load(open(on_screen_mfcc_path, "rb"))
            off_screen_mfcc = pkl.load(open(off_screen_mfcc_path, "rb"))
            if self.normalize_MFCC:
                mean = np.mean(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.std(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.where(std <= 1E-8, 1, std)
                on_screen_mfcc = (on_screen_mfcc - mean) / std  
                off_screen_mfcc = (off_screen_mfcc - mean) / std
                # now this is normalized to 0 mean and 1 std
            if self.pos_labels:
                on_screen_pos = pkl.load(open(on_screen_pos_path, "rb"))
                off_screen_pos = pkl.load(open(off_screen_pos_path, "rb")) 
            if on_screen_mfcc.shape[0] <= 50:
                continue
            # get input features
            input_features_on_screen = np.concatenate([on_screen_mfcc, on_screen_sentence_timing], axis=1)
            input_features_off_screen = np.concatenate([off_screen_mfcc, off_screen_sentence_timing], axis=1)
            if self.pos_labels: # the last 14 features are the POS tags
                input_features_on_screen = np.concatenate([input_features_on_screen, on_screen_pos], axis=1)
                input_features_off_screen = np.concatenate([input_features_off_screen, off_screen_pos], axis=1)
            input_feature = np.concatenate([input_features_on_screen, input_features_off_screen], axis=1)
            vel_output_target = dx_dt(output_aversion_label)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            self.input_features.append(input_feature)
            self.aversion_labels.append(output_aversion_label)
            self.velocity_labels.append(vel_output_target)
            counter += 1
    def __getitem__(self, idx):
        # pad all audio to 250 frames
        input_feature = self.input_features[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        aversion_label = self.aversion_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        velocity_label = self.velocity_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        # print(self.window_length, self.map[idx][1][0], self.map[idx][1][1], self.input_features[self.map[idx][0]].shape, self.aversion_labels[self.map[idx][0]].shape, self.velocity_labels[self.map[idx][0]].shape)
        if input_feature.shape[0] < self.window_length:
            missing_frames = self.window_length - input_feature.shape[0]
            padding = np.tile(np.expand_dims(self.filler_back, axis=0), [missing_frames, 1])
            input_feature = np.concatenate([input_feature, padding], axis=0)
            final_aversion_frame = aversion_label[-1]
            repeated_final_aversion_frame = np.tile(np.expand_dims(final_aversion_frame, axis=0), [missing_frames])
            aversion_label = np.concatenate([aversion_label, repeated_final_aversion_frame], axis=0)
            velocity_label = np.concatenate([velocity_label, np.zeros(missing_frames)], axis=0)  

        
        input_feature = torch.from_numpy(input_feature).double()
        if self.apply_time_mask and self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(self.time_mask(input_feature[:, 0:26]))
            input_feature[:, 46:72] = self.freq_mask(self.time_mask(input_feature[:, 46:72]))
        elif self.apply_time_mask:
            input_feature[:, 0:26] = self.time_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.time_mask(input_feature[:, 46:72])
        elif self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.freq_mask(input_feature[:, 46:72])
        aversion_label = torch.from_numpy(aversion_label).double()
        velocity_label = torch.from_numpy(velocity_label).double()
        if self.with_gaze:
            gaze = self.gaze_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
            return input_feature, [aversion_label, gaze, self.interlocutor_positions[self.map[idx][0]], velocity_label] 
        return input_feature, [aversion_label, velocity_label]      

class Runtime_parsing_Aversion_SelfTape111_with_word_vec(Dataset):
    def __init__(self, processed_data_path, videos_included=None, prev_dataset=None, pos_labels=True, long_aversion_only=False, shuffle=True, window_length=250, with_gaze=False, normalize_MFCC=False, apply_frequency_mask=False, apply_time_mask=False, word_vec_location="word_embedding_WavLM"):
        torch.set_default_tensor_type(torch.DoubleTensor)
        if prev_dataset is None:        
            self.data_root_path = processed_data_path
            self.shuffle = shuffle
            self.pos_labels = pos_labels
            self.window_length = window_length
            self.with_gaze = with_gaze
            self.long_aversion_only = long_aversion_only
            video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
            self.metadata = json.load(open(video_names_path, "r"))
            self.all_files_in_set = []
            if videos_included is None:
                videos_included = list(self.metadata.keys())
            self.all_files_in_set = videos_included
            self.gaussian_window = gaussian(5, 1)
            self.normalize_MFCC = normalize_MFCC
            self.apply_frequency_mask = apply_frequency_mask
            self.apply_time_mask = apply_time_mask
            self.word_vec_location = word_vec_location
            # load all input features and aversionl labels to memory
            self.input_features = []
            self.aversion_labels = []
            self.velocity_labels = []
            self.gaze_labels = []
            self.interlocutor_positions = []
            self.load_IO_features_to_memory()
            # generate a map to map the index of the dataset to the video
            self.map = {}
            self.dataset_length = 0
            self.parse_dataset()
            # generate filler for input features:
            self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0, 0.0])
            self.filler_back = np.concatenate([np.zeros(768), self.filler, np.zeros(6), np.zeros(768), self.filler, np.zeros(6)])
            if self.pos_labels:
                self.filler_back = np.concatenate([np.zeros(768), self.filler, np.zeros(20), np.zeros(768), self.filler, np.zeros(20)])
        else:
            self.data_root_path = prev_dataset.data_root_path
            self.shuffle = prev_dataset.shuffle
            self.pos_labels = prev_dataset.pos_labels
            self.window_length = prev_dataset.window_length
            self.window_length = window_length
            self.long_aversion_only = prev_dataset.long_aversion_only
            self.all_files_in_set = prev_dataset.all_files_in_set
            self.gaussian_window = prev_dataset.gaussian_window
            self.input_features = prev_dataset.input_features
            self.aversion_labels = prev_dataset.aversion_labels
            self.velocity_labels = prev_dataset.velocity_labels
            self.with_gaze = prev_dataset.with_gaze
            self.map = prev_dataset.map
            self.dataset_length = prev_dataset.dataset_length
            self.filler = prev_dataset.filler
            self.filler_back = prev_dataset.filler_back
            self.normalize_MFCC = prev_dataset.normalize_MFCC
            self.apply_time_mask = prev_dataset.apply_time_mask
            self.apply_frequency_mask = prev_dataset.apply_frequency_mask
            self.word_vec_location = prev_dataset.word_vec_location
            self.parse_dataset()
    def __len__(self):
        return self.dataset_length
    def parse_dataset(self):
        self.map = {}
        self.dataset_length = 0
        counter = 0
        for i in range(len(self.input_features)):
            # for randomly cutting the video
            random_offset = np.random.randint(0, self.window_length/2)
            # code starts here
            video_length = self.input_features[i].shape[0] - random_offset # if we start going through the video from the random offset, we will have this many frames left
            stride_length_video_per_segment = int(np.round(self.window_length/2))
            window_count = np.floor((video_length - (self.window_length - stride_length_video_per_segment)) / stride_length_video_per_segment)
            if self.input_features[i].shape[0] <= 25:
                continue
            if video_length <= 0:
                continue
            # add all the windows except the last window
            for w in range(0, int(window_count)):
                # start will be some offset away from the start
                video_window_start = stride_length_video_per_segment * w + random_offset
                video_window_end = video_window_start + self.window_length
                window_range = [video_window_start, video_window_end]
                self.map[counter] = [i, window_range]
                counter = counter + 1
            self.map[counter] = [i, [max(0, video_length-self.window_length), video_length]]
            counter += 1
        self.dataset_length = counter
    def time_mask(self, spec, T=30, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[0]
        for i in range(0, num_masks):
            # I only have 250 ish samples so I'm masking 20 max
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[t_zero:mask_end] = 0
            else: cloned[t_zero:mask_end] = cloned.mean()
        return cloned
    def freq_mask(self, spec, F=5, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
            else: cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned
    def load_IO_features_to_memory(self):
        for file_name in self.all_files_in_set:
            # get the aversion labels from the disk
            print(file_name)
            if self.long_aversion_only:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "long_aversion_label", file_name+".pkl"])
            else:
                output_aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", file_name+".pkl"])
            if self.with_gaze:
                gaze_label_path = os.path.join(*[self.data_root_path, "gaze", file_name+".pkl"])
                self.gaze_labels.append(pkl.load(open(gaze_label_path, "rb")))
                interlocutor_position_path = os.path.join(*["/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset/", "tinterlocutor_direction", file_name+".pkl"])
                self.interlocutor_positions.append(pkl.load(open(interlocutor_position_path, "rb")))
            output_aversion_label = pkl.load(open(output_aversion_label_path, "rb"))

            # get the input features from the disk 
            # on_screen_raw_audio_path = os.path.join
            on_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_0.pkl"]) 
            off_screen_sentence_timing_path = os.path.join(*[self.data_root_path, "sentence_timing", file_name+"_1.pkl"])
            on_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_0.pkl"])
            off_screen_mfcc_path = os.path.join(*[self.data_root_path, "audio", file_name+"_1.pkl"])
            on_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_0.pkl"])
            off_screen_pos_path = os.path.join(*[self.data_root_path, "word_POS", file_name+"_1.pkl"])
            # word vector path
            on_screen_word_vec_path = os.path.join(*[self.data_root_path, self.word_vec_location, file_name+"_0.pkl"])
            off_screen_word_vec_path = os.path.join(*[self.data_root_path, self.word_vec_location, file_name+"_1.pkl"])
            
            # load the input features from the disk
            on_screen_sentence_timing = pkl.load(open(on_screen_sentence_timing_path, "rb"))
            off_screen_sentence_timing = pkl.load(open(off_screen_sentence_timing_path, "rb"))
            on_screen_mfcc = pkl.load(open(on_screen_mfcc_path, "rb"))
            off_screen_mfcc = pkl.load(open(off_screen_mfcc_path, "rb"))
            if self.normalize_MFCC:
                mean = np.mean(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.std(on_screen_mfcc + off_screen_mfcc, axis=0)
                std = np.where(std <= 1E-8, 1, std)
                on_screen_mfcc = (on_screen_mfcc - mean) / std  
                off_screen_mfcc = (off_screen_mfcc - mean) / std
                # now this is normalized to 0 mean and 1 std
            if self.pos_labels:
                on_screen_pos = pkl.load(open(on_screen_pos_path, "rb"))
                off_screen_pos = pkl.load(open(off_screen_pos_path, "rb")) 
            if on_screen_mfcc.shape[0] <= 50:
                continue
            # get input features
            on_sreen_input_features_word_embedding = pkl.load(open(on_screen_word_vec_path, "rb"))
            off_sreen_input_features_word_embedding = pkl.load(open(off_screen_word_vec_path, "rb"))
            input_features_on_screen = np.concatenate([on_sreen_input_features_word_embedding, on_screen_mfcc, on_screen_sentence_timing], axis=1)
            input_features_off_screen = np.concatenate([off_sreen_input_features_word_embedding, off_screen_mfcc, off_screen_sentence_timing], axis=1)
            if self.pos_labels: # the last 14 features are the POS tags
                input_features_on_screen = np.concatenate([input_features_on_screen, on_screen_pos], axis=1)
                input_features_off_screen = np.concatenate([input_features_off_screen, off_screen_pos], axis=1)
            input_feature = np.concatenate([input_features_on_screen, input_features_off_screen], axis=1)
            vel_output_target = dx_dt(output_aversion_label)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            self.input_features.append(input_feature)
            self.aversion_labels.append(output_aversion_label)
            self.velocity_labels.append(vel_output_target)
    def __getitem__(self, idx):
        # pad all audio to 250 frames
        input_feature = self.input_features[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        aversion_label = self.aversion_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        velocity_label = self.velocity_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
        # print(self.window_length, self.map[idx][1][0], self.map[idx][1][1], self.input_features[self.map[idx][0]].shape, self.aversion_labels[self.map[idx][0]].shape, self.velocity_labels[self.map[idx][0]].shape)
        if input_feature.shape[0] < self.window_length:
            missing_frames = self.window_length - input_feature.shape[0]
            padding = np.tile(np.expand_dims(self.filler_back, axis=0), [missing_frames, 1])
            input_feature = np.concatenate([input_feature, padding], axis=0)
            final_aversion_frame = aversion_label[-1]
            repeated_final_aversion_frame = np.tile(np.expand_dims(final_aversion_frame, axis=0), [missing_frames])
            aversion_label = np.concatenate([aversion_label, repeated_final_aversion_frame], axis=0)
            velocity_label = np.concatenate([velocity_label, np.zeros(missing_frames)], axis=0)  

        
        input_feature = torch.from_numpy(input_feature).double()
        if self.apply_time_mask and self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(self.time_mask(input_feature[:, 0:26]))
            input_feature[:, 46:72] = self.freq_mask(self.time_mask(input_feature[:, 46:72]))
        elif self.apply_time_mask:
            input_feature[:, 0:26] = self.time_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.time_mask(input_feature[:, 46:72])
        elif self.apply_frequency_mask:
            input_feature[:, 0:26] = self.freq_mask(input_feature[:, 0:26])
            input_feature[:, 46:72] = self.freq_mask(input_feature[:, 46:72])
        aversion_label = torch.from_numpy(aversion_label).double()
        velocity_label = torch.from_numpy(velocity_label).double()
        if self.with_gaze:
            gaze = self.gaze_labels[self.map[idx][0]][self.map[idx][1][0]:self.map[idx][1][1]]
            return input_feature, [aversion_label, gaze, self.interlocutor_positions[self.map[idx][0]], velocity_label] 
        return input_feature, [aversion_label, velocity_label]      
  