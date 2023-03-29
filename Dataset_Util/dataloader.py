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
import soundfile as sf
# import utility functions
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Desktop/openpose/python/')
# sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox")

from Geometry_Util import rotation_angles_frome_positions

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
class  SegmentDataset_SelfTape111(Dataset):
    def __init__(self, processed_data_path, win_length=10, stride_length=5):
        # save dataset root path
        self.data_root_path = processed_data_path
        self.count = 0
        # load video names
        video_names_path = os.path.join(*[processed_data_path, "metadata.json"])
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
        aversion = pkl.load(open(output_aversion_path, "rb"))

        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        audio_onscreen = audio_onscreen[a_range[0]:a_range[1]]
        audio_offscreen = audio_offscreen[a_range[0]:a_range[1]]
        return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks, aversion]  
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
        # aversion = pkl.load(open(output_aversion_path, "rb"))

        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        shot_range = self.video_metadata[idx]["video_range"]
        return [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks], [file_name, shot_range], 